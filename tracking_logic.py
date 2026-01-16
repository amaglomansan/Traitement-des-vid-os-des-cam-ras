
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os

class ObjectTrackingSystem:
    def __init__(self, model_weight='yolov8n.pt'):
        self.model = YOLO(model_weight)
        self.results_data = []

    def is_inside_zone(self, point, zone_poly):
        if zone_poly is None:
            return False
        return cv2.pointPolygonTest(np.array(zone_poly, np.int32), point, False) >= 0

    def process_video(self, camera_name, video_path, zone_poly=None, output_folder='output'):
        """Performs detection and tracking on a single video file."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_path = os.path.join(output_folder, f"{camera_name}_tracked.mp4")
        csv_path = os.path.join(output_folder, f"{camera_name}_results.csv")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_count = 0
        camera_detections = []
        
        print(f"Traitement de {camera_name} ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} images)...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run tracking with YOLOv8
            # Note: IDs are unique within this video session
            results = self.model.track(frame, persist=True, verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                colors = results[0].boxes.cls.cpu().numpy().astype(int)
                names = self.model.names

                for box, obj_id, cls_idx in zip(boxes, ids, colors):
                    x1, y1, x2, y2 = box
                    w_box = int(x2 - x1)
                    h_box = int(y2 - y1)
                    xc = int(x1 + w_box / 2)
                    yc = int(y1 + h_box / 2)
                    
                    obj_class = names[cls_idx]
                    in_alert = False
                    
                    if zone_poly:
                        in_alert = self.is_inside_zone((xc, yc), zone_poly)
                        color = (0, 0, 255) if in_alert else (0, 255, 0)
                    else:
                        color = (0, 255, 0)

                    # Annotation
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"{obj_class} ID:{obj_id}", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if in_alert:
                        cv2.putText(frame, "ZONE ALERTE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # Save data for this frame
                    detection = {
                        'camera': camera_name,
                        'frame': frame_count,
                        'timestamp': round(frame_count / fps, 2),
                        'obj_id': obj_id,
                        'class': obj_class,
                        'bbox': [int(x1), int(y1), w_box, h_box],
                        'x_c': xc,
                        'y_c': yc,
                        'w': w_box,
                        'h': h_box,
                        'in_alert_zone': in_alert,
                        'v_width': width,
                        'v_height': height
                    }
                    self.results_data.append(detection)
                    camera_detections.append(detection)

            if zone_poly:
                cv2.polylines(frame, [np.array(zone_poly, np.int32)], True, (255, 255, 0), 2)

            out.write(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  > Image {frame_count} traitée...", end='\r')
            
        cap.release()
        out.release()
        
        if camera_detections:
            pd.DataFrame(camera_detections).to_csv(csv_path, index=False)
            print(f"\n{camera_name} terminé. {len(pd.DataFrame(camera_detections)['obj_id'].unique())} objets détectés.")
            
        return out_path

    def load_existing_results(self, folder):
        """Recharge les données CSV du dossier spécifié."""
        if not os.path.exists(folder):
            print(f"Dossier {folder} introuvable.")
            return
        
        self.results_data = [] # Reset to avoid duplicates
        count = 0
        for file in os.listdir(folder):
            if file.endswith("_results.csv"):
                df = pd.read_csv(os.path.join(folder, file))
                self.results_data.extend(df.to_dict('records'))
                count += 1
        print(f"Chargement terminé : {count} fichiers de données indexés ({len(self.results_data)} détections).")

    def get_dataframe(self):
        return pd.DataFrame(self.results_data)

    def analyze_trajectories(self):
        """Calcule les statistiques de trajectoire par objet et par caméra."""
        df = self.get_dataframe()
        if df.empty:
            return pd.DataFrame()
            
        summary = []
        for (camera, obj_id), group in df.groupby(['camera', 'obj_id']):
            group = group.sort_values('timestamp')
            
            start_time = group['timestamp'].min()
            end_time = group['timestamp'].max()
            duration = round(end_time - start_time, 2)
            
            entry_pos = (group.iloc[0]['x_c'], group.iloc[0]['y_c'])
            exit_pos = (group.iloc[-1]['x_c'], group.iloc[-1]['y_c'])
            
            v_w = group.iloc[0]['v_width'] if 'v_width' in group.columns else 640
            v_h = group.iloc[0]['v_height'] if 'v_height' in group.columns else 480

            # Direction simplifiée
            dx = exit_pos[0] - entry_pos[0]
            dy = exit_pos[1] - entry_pos[1]
            if abs(dx) > abs(dy):
                direction = "Droite" if dx > 0 else "Gauche"
            else:
                direction = "Bas" if dy > 0 else "Haut"
            if abs(dx) < 20 and abs(dy) < 20:
                direction = "Statique"

            # État : Actif/Sorti/Disparu
            # Si le dernier point est proche des bords, on dit "Sorti"
            # Sinon, s'il disparaît au centre de l'image, on dit "Disparu"
            margin = 50
            is_near_edge = (exit_pos[0] < margin or exit_pos[0] > (v_w - margin) or 
                            exit_pos[1] < margin or exit_pos[1] > (v_h - margin))
            
            state = "Sorti" if is_near_edge else "Disparu"

            summary.append({
                'camera': camera,
                'obj_id': obj_id,
                'class': group.iloc[0]['class'],
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'entry_point': entry_pos,
                'exit_point': exit_pos,
                'direction': direction,
                'state': state,
                'alert_count': group['in_alert_zone'].sum()
            })
            
        return pd.DataFrame(summary).sort_values(['camera', 'start_time'])

    def get_multi_camera_path(self, target_id, target_class='person'):
        """
        Reconstruit le parcours complet d'un objet. 
        Note: Actuellement basé sur l'ID (unique par vidéo). 
        Pour un vrai tracking multi-caméra, il faudrait une étape de Re-ID.
        """
        df_summary = self.analyze_trajectories()
        if df_summary.empty:
            return None
            
        # On filtre par ID. Attention : l'ID change d'une vidéo à l'autre normalement.
        # Ici on suppose que l'utilisateur veut voir le parcours d'un ID s'il est constant
        # ou on laisse l'utilisateur choisir les IDs à lier.
        path = df_summary[df_summary['obj_id'] == target_id].sort_values('start_time')
        
        if path.empty:
            return None
            
        return path

