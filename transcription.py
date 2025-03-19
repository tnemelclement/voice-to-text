import whisper
import argparse
import torch
import os
from ollama import Client
import ssl
from typing import Dict, Any

# Configuration
MODEL_WHISPER = "base"  # Modèle Whisper à utiliser
MODEL_OLLAMA = "mistral"  # Modèle Ollama pour le post-traitement (optionnel)

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

def analyze_transcription(result: Dict[str, Any], output_file: str = None, use_ollama: bool = False) -> str:
    """
    Analyse et traite le résultat de la transcription
    
    Args:
        result (Dict[str, Any]): Résultat de la transcription Whisper
        output_file (str, optional): Fichier de sortie pour le texte
        use_ollama (bool): Utiliser Ollama pour post-traiter la transcription
        
    Returns:
        str: Texte final de la transcription
    """
    transcription = result["text"]
    
    if use_ollama:
        print("Post-traitement avec Ollama...")
        client = Client(host="http://localhost:11434")
        prompt = f"Voici une transcription audio qui peut contenir des erreurs. Corrige-la pour qu'elle soit plus lisible:\n\n{transcription}"
        response = client.chat(model=MODEL_OLLAMA, messages=[
            {"role": "user", "content": prompt}
        ])
        transcription = response["message"]["content"]
    
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription)
        print(f"Transcription enregistrée dans: {output_file}")
    else:
        print("\nTranscription:")
        print(transcription)
    
    return transcription

def main():
    parser = argparse.ArgumentParser(description="Outils de transcription et d'analyse audio")
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")
    
    # Commande de transcription
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcrire un fichier audio")
    transcribe_parser.add_argument("audio_file", help="Chemin vers le fichier audio à transcrire")
    transcribe_parser.add_argument("-o", "--output", help="Fichier de sortie pour la transcription")
    
    # Commande d'analyse
    analyze_parser = subparsers.add_parser("analyze", help="Analyser une transcription existante")
    analyze_parser.add_argument("input_file", help="Fichier contenant la transcription à analyser")
    analyze_parser.add_argument("--ollama", action="store_true", help="Utiliser Ollama pour post-traiter la transcription")
    
    # Commande complète
    full_parser = subparsers.add_parser("full", help="Transcrire et analyser un fichier audio")
    full_parser.add_argument("audio_file", help="Chemin vers le fichier audio à transcrire")
    full_parser.add_argument("-o", "--output", help="Fichier de sortie pour la transcription")
    full_parser.add_argument("--ollama", action="store_true", help="Utiliser Ollama pour post-traiter la transcription")
    
    args = parser.parse_args()
    
    if args.command == "transcribe":
        if not os.path.exists(args.audio_file):
            print(f"Erreur: Le fichier {args.audio_file} n'existe pas.")
            return
        result = transcribe_audio(args.audio_file)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result["text"])
            print(f"Transcription enregistrée dans: {args.output}")
    
    elif args.command == "analyze":
        if not os.path.exists(args.input_file):
            print(f"Erreur: Le fichier {args.input_file} n'existe pas.")
            return
        with open(args.input_file, "r", encoding="utf-8") as f:
            transcription = f.read()
        result = {"text": transcription}
        analyze_transcription(result, use_ollama=args.ollama)
    
    elif args.command == "full":
        if not os.path.exists(args.audio_file):
            print(f"Erreur: Le fichier {args.audio_file} n'existe pas.")
            return
        result = transcribe_audio(args.audio_file)
        analyze_transcription(result, args.output, args.ollama)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
