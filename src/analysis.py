from typing import Dict, Any, List
from ollama import Client
from .config import MODEL_OLLAMA, OLLAMA_HOST, PROMPT_CORRECTION, PROMPT_QUESTION

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
        client = Client(host=OLLAMA_HOST)
        prompt = PROMPT_CORRECTION.format(transcription=transcription)
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

def ask_question(transcription: str, question: str) -> str:
    """
    Pose une question sur la transcription
    
    Args:
        transcription (str): Texte de la transcription
        question (str): Question à poser
        
    Returns:
        str: Réponse à la question
    """
    client = Client(host=OLLAMA_HOST)
    prompt = PROMPT_QUESTION.format(transcription=transcription, question=question)
    
    print(f"\nQuestion: {question}")
    response = client.chat(model=MODEL_OLLAMA, messages=[
        {"role": "user", "content": prompt}
    ])
    
    return response["message"]["content"]

def generate_summary(transcription: str) -> str:
    """
    Génère un résumé de la transcription
    
    Args:
        transcription (str): Texte de la transcription
        
    Returns:
        str: Résumé de la transcription
    """
    client = Client(host=OLLAMA_HOST)
    prompt = f"Fais un résumé concis de cette transcription:\n\n{transcription}"
    
    print("\nGénération du résumé...")
    response = client.chat(model=MODEL_OLLAMA, messages=[
        {"role": "user", "content": prompt}
    ])
    
    return response["message"]["content"]

def extract_keywords(transcription: str) -> List[str]:
    """
    Extrait les mots-clés principaux de la transcription
    
    Args:
        transcription (str): Texte de la transcription
        
    Returns:
        List[str]: Liste des mots-clés
    """
    client = Client(host=OLLAMA_HOST)
    prompt = f"Extrais les 5 mots-clés principaux de cette transcription. Réponds uniquement avec les mots-clés, séparés par des virgules:\n\n{transcription}"
    
    print("\nExtraction des mots-clés...")
    response = client.chat(model=MODEL_OLLAMA, messages=[
        {"role": "user", "content": prompt}
    ])
    
    return [kw.strip() for kw in response["message"]["content"].split(",")]

def analyze_sentiment(transcription: str) -> str:
    """
    Analyse le sentiment général de la transcription
    
    Args:
        transcription (str): Texte de la transcription
        
    Returns:
        str: Analyse du sentiment
    """
    client = Client(host=OLLAMA_HOST)
    prompt = f"Analyse le sentiment général de cette transcription (positif, négatif, neutre) et explique pourquoi:\n\n{transcription}"
    
    print("\nAnalyse du sentiment...")
    response = client.chat(model=MODEL_OLLAMA, messages=[
        {"role": "user", "content": prompt}
    ])
    
    return response["message"]["content"] 