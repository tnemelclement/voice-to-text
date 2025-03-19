import os
import torch
import whisper
import soundfile as sf
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering
from datetime import timedelta
import warnings
import subprocess
import tempfile
warnings.filterwarnings("ignore")

def convert_m4a_to_wav(m4a_file):
    """
    Convertit un fichier M4A en WAV en utilisant ffmpeg.
    
    Args:
        m4a_file (str): Chemin vers le fichier M4A
        
    Returns:
        str: Chemin vers le fichier WAV temporaire
    """
    # Créer un fichier temporaire pour le WAV
    temp_dir = tempfile.gettempdir()
    wav_file = os.path.join(temp_dir, "temp_audio.wav")
    
    # Convertir M4A en WAV avec ffmpeg
    cmd = [
        "ffmpeg",
        "-i", m4a_file,
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        wav_file,
        "-y"  # Écraser le fichier s'il existe
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return wav_file
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la conversion audio: {e.stderr.decode()}")
        return None

def format_timestamp(seconds):
    """Convertit les secondes en format timestamp lisible"""
    td = timedelta(seconds=seconds)
    minutes, seconds = divmod(td.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def transcribe_with_speaker_diarization(audio_file, output_file, num_speakers=None):
    """
    Transcrit un fichier audio en identifiant les différents locuteurs.
    
    Args:
        audio_file (str): Chemin vers le fichier audio
        output_file (str): Fichier de sortie pour le dialogue
        num_speakers (int, optional): Nombre de locuteurs à détecter (si connu)
    """
    print("Chargement des modèles...")
    
    # Charger le modèle Whisper pour la transcription
    try:
        whisper_model = whisper.load_model("medium")
        print("Modèle Whisper chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Whisper: {e}")
        return
    
    # Charger le modèle SpeechBrain pour l'extraction d'embeddings vocaux
    try:
        spk_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        )
        print("Modèle SpeechBrain chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle SpeechBrain: {e}")
        return
    
    # Étape 1: Transcrire l'audio avec Whisper pour obtenir les segments
    print(f"Transcription de l'audio: {audio_file}")
    result = whisper_model.transcribe(audio_file, word_timestamps=True)
    segments = result["segments"]
    
    # Étape 2: Extraire les embeddings vocaux pour chaque segment
    print("Extraction des caractéristiques vocales...")
    embeddings = []
    segment_info = []
    
    # Convertir M4A en WAV si nécessaire
    if audio_file.lower().endswith('.m4a'):
        wav_file = convert_m4a_to_wav(audio_file)
        if wav_file is None:
            print("Échec de la conversion audio")
            return
        audio_file = wav_file
    
    # Charger l'audio avec soundfile
    try:
        signal, fs = sf.read(audio_file)
        if len(signal.shape) > 1:  # Convertir stéréo en mono si nécessaire
            signal = np.mean(signal, axis=1)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier audio: {e}")
        return
    
    # Normaliser le signal audio
    signal = signal / np.max(np.abs(signal))
    
    for segment in segments:
        start_time = segment["start"]
        end_time = segment["end"]
        
        # Extraire le segment audio
        start_frame = int(start_time * fs)
        end_frame = int(end_time * fs)
        
        if end_frame <= start_frame:
            continue
        
        segment_audio = signal[start_frame:end_frame]
        
        # Vérifier si le segment est assez long
        if len(segment_audio) < 0.5 * fs:  # Moins de 0.5 seconde
            continue
        
        # Convertir en tensor pour SpeechBrain
        segment_tensor = torch.FloatTensor(segment_audio).unsqueeze(0)
        
        # Extraire l'embedding
        with torch.no_grad():
            try:
                emb = spk_model.encode_batch(segment_tensor)
                emb = emb.squeeze().cpu().numpy()
                embeddings.append(emb)
                segment_info.append({
                    "start": start_time,
                    "end": end_time,
                    "text": segment["text"]
                })
            except Exception as e:
                print(f"Erreur lors de l'extraction d'embedding: {e}")
                continue
    
    if len(embeddings) == 0:
        print("Aucun embedding extrait. Vérifiez l'audio.")
        return
    
    # Étape 3: Clustering pour identifier les locuteurs
    print("Identification des locuteurs...")
    embeddings_array = np.vstack(embeddings)
    
    # Déterminer le nombre de locuteurs si non spécifié
    if num_speakers is None:
        # Estimation automatique du nombre de locuteurs
        from sklearn.metrics import silhouette_score
        
        range_n_clusters = range(2, min(len(embeddings_array), 10))
        best_score = -1
        best_n_clusters = 2
        
        for n_clusters in range_n_clusters:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(embeddings_array)
            
            if len(set(cluster_labels)) <= 1:
                continue
                
            try:
                score = silhouette_score(embeddings_array, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except:
                continue
        
        num_speakers = best_n_clusters
        print(f"Nombre estimé de locuteurs: {num_speakers}")
    
    # Clustering pour identifier les locuteurs
    clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings_array)
    labels = clustering.labels_
    
    # Étape 4: Attribuer les locuteurs aux segments
    for i, segment in enumerate(segment_info):
        if i < len(labels):
            segment["speaker"] = f"SPEAKER_{labels[i]}"
    
    # Étape 5: Fusionner les segments consécutifs du même locuteur
    merged_segments = []
    current_speaker = None
    current_text = ""
    current_start = 0
    
    for segment in segment_info:
        if segment["speaker"] != current_speaker:
            if current_speaker:
                merged_segments.append({
                    "speaker": current_speaker,
                    "start": current_start,
                    "end": segment["start"],
                    "text": current_text.strip()
                })
            current_speaker = segment["speaker"]
            current_text = segment["text"]
            current_start = segment["start"]
        else:
            current_text += " " + segment["text"]
    
    # Ajouter le dernier segment
    if current_speaker and segment_info:
        merged_segments.append({
            "speaker": current_speaker,
            "start": current_start,
            "end": segment_info[-1]["end"],
            "text": current_text.strip()
        })
    
    # Étape 6: Écrire le dialogue dans un fichier
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in merged_segments:
            speaker_name = segment["speaker"]
            timestamp = format_timestamp(segment["start"])
            f.write(f"[{timestamp}] {speaker_name}: {segment['text']}\n\n")
    
    print(f"Dialogue transcrit enregistré dans: {output_file}")
    print(f"Nombre de locuteurs détectés: {num_speakers}")
    
    # Nettoyer le fichier temporaire WAV si nécessaire
    if audio_file.endswith("temp_audio.wav"):
        try:
            os.remove(audio_file)
        except:
            pass 