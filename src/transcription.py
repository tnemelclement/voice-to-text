import whisper
import ssl
from typing import Dict, Any
from .config import MODEL_WHISPER

# Solution pour le problème de certificat SSL sur macOS
ssl._create_default_https_context = ssl._create_unverified_context

def transcribe_audio(audio_file: str) -> Dict[str, Any]:
    """
    Transcrit un fichier audio en utilisant Whisper
    
    Args:
        audio_file (str): Chemin vers le fichier audio
        
    Returns:
        Dict[str, Any]: Résultat de la transcription contenant le texte et les métadonnées
    """
    print(f"Chargement du modèle Whisper {MODEL_WHISPER}...")
    model = whisper.load_model(MODEL_WHISPER)
    
    print(f"Transcription du fichier audio: {audio_file}")
    result = model.transcribe(audio_file)
    
    return result 