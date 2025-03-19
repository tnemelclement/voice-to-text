# Configuration des modèles
MODEL_WHISPER = "base"  # Modèle Whisper à utiliser
MODEL_OLLAMA = "mistral"  # Modèle Ollama pour le post-traitement

# Configuration du serveur Ollama
OLLAMA_HOST = "http://localhost:11434"

# Configuration des prompts
PROMPT_CORRECTION = "Voici une transcription audio qui peut contenir des erreurs. Corrige-la pour qu'elle soit plus lisible:\n\n{transcription}"
PROMPT_QUESTION = "En te basant sur cette transcription, réponds à la question suivante:\n\nTranscription:\n{transcription}\n\nQuestion: {question}" 