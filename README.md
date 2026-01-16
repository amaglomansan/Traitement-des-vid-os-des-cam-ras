# Projet de Détection et de Tracking d'Objets Multi-Caméras

Ce projet implémente un système de surveillance intelligent capable de détecter, tracker et analyser les trajectoires d'objets (personnes, véhicules, etc.) à travers plusieurs caméras.

## Fonctionnalités
- **Détection et Classement** : Utilisation de YOLOv8 pour identifier les objets.
- **Single Target Tracking (STT)** : Suivi individuel des objets avec ID persistants par caméra.
- **Analyse de Trajectoire** : Extraction des points d'entrée, de sortie et du temps de présence.
- **Zones d'Alerte** : Définition de zones sensibles déclenchant des notifications visuelles.
- **Multi-Caméra** : Cartographie des caméras et suivi de l'objet à travers le réseau de surveillance.

## Structure du Projet
- `detection_tracking.ipynb` : Notebook principal contenant l'analyse et la visualisation.
- `tracking_logic.py` : Moteur de tracking et de traitement vidéo.
- `config.py` : Configuration des caméras et des zones d'alerte.
- `requirements.txt` : Liste des dépendances Python.
- `video/` : Dossier contenant les sources vidéos originales.
- `output/` : Dossier contenant les vidéos traitées avec annotations.

## Installation
1. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Lancez le notebook Jupyter :
   ```bash
   jupyter notebook detection_tracking.ipynb
   ```
   
