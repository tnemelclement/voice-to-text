import argparse
import os
from typing import Optional
from .transcription import transcribe_audio
from .analysis import (
    analyze_transcription, 
    ask_question, 
    generate_summary, 
    extract_keywords, 
    analyze_sentiment
)
from .diarization import transcribe_with_speaker_diarization

def setup_parser() -> argparse.ArgumentParser:
    """Configure et retourne le parser d'arguments"""
    parser = argparse.ArgumentParser(description="Outils de transcription et d'analyse audio")
    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")
    
    # Commande de transcription
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcrire un fichier audio")
    transcribe_parser.add_argument("audio_file", help="Chemin vers le fichier audio à transcrire")
    transcribe_parser.add_argument("-o", "--output", help="Fichier de sortie pour la transcription")
    
    # Commande de diarisation
    diarize_parser = subparsers.add_parser("diarize", help="Transcrire un fichier audio avec identification des locuteurs")
    diarize_parser.add_argument("audio_file", help="Chemin vers le fichier audio à transcrire")
    diarize_parser.add_argument("-o", "--output", help="Fichier de sortie pour le dialogue", default="dialogue.txt")
    diarize_parser.add_argument("-s", "--speakers", type=int, help="Nombre de locuteurs (si connu)", default=None)
    
    # Commande d'analyse
    analyze_parser = subparsers.add_parser("analyze", help="Analyser une transcription existante")
    analyze_parser.add_argument("input_file", help="Fichier contenant la transcription à analyser")
    analyze_parser.add_argument("--ollama", action="store_true", help="Utiliser Ollama pour post-traiter la transcription")
    
    # Commande complète
    full_parser = subparsers.add_parser("full", help="Transcrire et analyser un fichier audio")
    full_parser.add_argument("audio_file", help="Chemin vers le fichier audio à transcrire")
    full_parser.add_argument("-o", "--output", help="Fichier de sortie pour la transcription")
    full_parser.add_argument("--ollama", action="store_true", help="Utiliser Ollama pour post-traiter la transcription")
    
    # Commande de questions
    question_parser = subparsers.add_parser("ask", help="Poser une question sur une transcription")
    question_parser.add_argument("input_file", help="Fichier contenant la transcription")
    question_parser.add_argument("question", help="Question à poser sur la transcription")
    
    # Commande de résumé
    summary_parser = subparsers.add_parser("summarize", help="Générer un résumé de la transcription")
    summary_parser.add_argument("input_file", help="Fichier contenant la transcription")
    
    # Commande de mots-clés
    keywords_parser = subparsers.add_parser("keywords", help="Extraire les mots-clés de la transcription")
    keywords_parser.add_argument("input_file", help="Fichier contenant la transcription")
    
    # Commande d'analyse de sentiment
    sentiment_parser = subparsers.add_parser("sentiment", help="Analyser le sentiment de la transcription")
    sentiment_parser.add_argument("input_file", help="Fichier contenant la transcription")
    
    return parser

def read_transcription(file_path: str) -> str:
    """Lit le contenu d'un fichier de transcription"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def handle_transcribe(args: argparse.Namespace) -> None:
    """Gère la commande de transcription"""
    if not os.path.exists(args.audio_file):
        print(f"Erreur: Le fichier {args.audio_file} n'existe pas.")
        return
    result = transcribe_audio(args.audio_file)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"Transcription enregistrée dans: {args.output}")

def handle_analyze(args: argparse.Namespace) -> None:
    """Gère la commande d'analyse"""
    if not os.path.exists(args.input_file):
        print(f"Erreur: Le fichier {args.input_file} n'existe pas.")
        return
    transcription = read_transcription(args.input_file)
    result = {"text": transcription}
    analyze_transcription(result, use_ollama=args.ollama)

def handle_full(args: argparse.Namespace) -> None:
    """Gère la commande complète"""
    if not os.path.exists(args.audio_file):
        print(f"Erreur: Le fichier {args.audio_file} n'existe pas.")
        return
    result = transcribe_audio(args.audio_file)
    analyze_transcription(result, args.output, args.ollama)

def handle_ask(args: argparse.Namespace) -> None:
    """Gère la commande de questions"""
    if not os.path.exists(args.input_file):
        print(f"Erreur: Le fichier {args.input_file} n'existe pas.")
        return
    transcription = read_transcription(args.input_file)
    answer = ask_question(transcription, args.question)
    print("\nRéponse:")
    print(answer)

def handle_summarize(args: argparse.Namespace) -> None:
    """Gère la commande de résumé"""
    if not os.path.exists(args.input_file):
        print(f"Erreur: Le fichier {args.input_file} n'existe pas.")
        return
    transcription = read_transcription(args.input_file)
    summary = generate_summary(transcription)
    print("\nRésumé:")
    print(summary)

def handle_keywords(args: argparse.Namespace) -> None:
    """Gère la commande de mots-clés"""
    if not os.path.exists(args.input_file):
        print(f"Erreur: Le fichier {args.input_file} n'existe pas.")
        return
    transcription = read_transcription(args.input_file)
    keywords = extract_keywords(transcription)
    print("\nMots-clés:")
    print(", ".join(keywords))

def handle_sentiment(args: argparse.Namespace) -> None:
    """Gère la commande d'analyse de sentiment"""
    if not os.path.exists(args.input_file):
        print(f"Erreur: Le fichier {args.input_file} n'existe pas.")
        return
    transcription = read_transcription(args.input_file)
    sentiment = analyze_sentiment(transcription)
    print("\nAnalyse du sentiment:")
    print(sentiment)

def handle_diarize(args: argparse.Namespace) -> None:
    """Gère la commande de diarisation"""
    if not os.path.exists(args.audio_file):
        print(f"Erreur: Le fichier {args.audio_file} n'existe pas.")
        return
    transcribe_with_speaker_diarization(args.audio_file, args.output, args.speakers)

def main() -> None:
    """Point d'entrée principal du CLI"""
    parser = setup_parser()
    args = parser.parse_args()
    
    handlers = {
        "transcribe": handle_transcribe,
        "analyze": handle_analyze,
        "full": handle_full,
        "ask": handle_ask,
        "summarize": handle_summarize,
        "keywords": handle_keywords,
        "sentiment": handle_sentiment,
        "diarize": handle_diarize
    }
    
    if args.command in handlers:
        handlers[args.command](args)
    else:
        parser.print_help() 