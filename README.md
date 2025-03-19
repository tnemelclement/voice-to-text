# Transcription Audio vers Texte

Un outil en ligne de commande pour transcrire des fichiers audio en texte, avec option de post-traitement intelligent et possibilité de poser des questions sur la transcription.

## Prérequis

- Python 3.8 ou supérieur
- ffmpeg (pour la conversion audio)
- Ollama (pour l'analyse avancée)

## Installation sur macOS

### 1. Installation de Homebrew

Si vous n'avez pas encore Homebrew, installez-le :

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Installation des dépendances système

```bash
# Installation de ffmpeg
brew install ffmpeg

# Installation de Python (si pas déjà installé)
brew install python

# Installation de Ollama
brew install ollama
```

### 3. Configuration de l'environnement Python

```bash
# Création d'un environnement virtuel
python -m venv venv

# Activation de l'environnement virtuel
source venv/bin/activate

# Installation des dépendances Python
pip install -r requirements.txt
```

### 4. Téléchargement des modèles

Les modèles nécessaires seront téléchargés automatiquement lors de la première utilisation. Cela peut prendre quelques minutes selon votre connexion internet.

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/votre-username/voice-to-text.git
cd voice-to-text
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. (Optionnel) Installez Ollama pour le post-traitement et les questions :
- Suivez les instructions d'installation sur [le site officiel d'Ollama](https://ollama.ai)
- Assurez-vous que le serveur Ollama est en cours d'exécution (`ollama serve`)

## Utilisation

Le script propose huit commandes principales :

### 1. Transcription simple

Pour transcrire un fichier audio en texte :
```bash
python -m src transcribe chemin/vers/audio.m4a -o transcription.txt
```

### 2. Transcription avec identification des locuteurs

Pour transcrire un fichier audio en identifiant les différents locuteurs :
```bash
python -m src diarize chemin/vers/audio.m4a -o dialogue.txt
```

Options disponibles :
- `-o, --output` : Fichier de sortie pour le dialogue (par défaut : "dialogue.txt")
- `-s, --speakers` : Nombre de locuteurs (si connu)

### 3. Analyse d'une transcription existante

Pour analyser et améliorer une transcription existante :
```bash
python -m src analyze transcription.txt --ollama
```

### 4. Transcription et analyse complète

Pour transcrire et analyser en une seule commande :
```bash
python -m src full chemin/vers/audio.m4a -o transcription.txt --ollama
```

### 5. Poser une question sur une transcription

Pour poser une question sur une transcription existante :
```bash
python -m src ask transcription.txt "Quel est le sujet principal de cette transcription ?"
```

### 6. Générer un résumé

Pour obtenir un résumé concis de la transcription :
```bash
python -m src summarize transcription.txt
```

### 7. Extraire les mots-clés

Pour extraire les mots-clés principaux :
```bash
python -m src keywords transcription.txt
```

### 8. Analyser le sentiment

Pour analyser le sentiment général de la transcription :
```bash
python -m src sentiment transcription.txt
```

## Options disponibles

- `-o, --output` : Spécifier le fichier de sortie pour la transcription
- `--ollama` : Utiliser Ollama pour le post-traitement intelligent du texte

## Formats audio supportés

Le script utilise Whisper qui supporte les formats audio suivants :
- MP3
- WAV
- M4A
- OGG
- FLAC
- Et autres formats courants

## Configuration

Vous pouvez modifier les paramètres dans le fichier `src/config.py` :
- `MODEL_WHISPER` : Modèle Whisper à utiliser (par défaut : "base")
- `MODEL_OLLAMA` : Modèle Ollama pour le post-traitement (par défaut : "mistral")
- `OLLAMA_HOST` : Adresse du serveur Ollama
- `PROMPT_CORRECTION` : Template pour la correction du texte
- `PROMPT_QUESTION` : Template pour les questions

## Structure du projet

```
voice-to-text/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── transcription.py
│   ├── analysis.py
│   └── diarization.py
├── requirements.txt
└── README.md
```

## Aide

Pour voir l'aide et toutes les options disponibles :
```bash
python -m src --help
```

Pour l'aide d'une commande spécifique :
```bash
python -m src [commande] --help
```

## Licence

MIT 