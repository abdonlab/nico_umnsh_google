# ---------------------------------------
# speech_utils.py
# Módulo simple para voz masculina/neutral
# No cambia nada del resto de tu app
# ---------------------------------------

import streamlit as st
from google.cloud import texttospeech
from google.oauth2 import service_account

def _get_client():
    # Usa credenciales desde st.secrets o desde el entorno
    if "service_account" in st.secrets:
        info = dict(st.secrets["service_account"])
        creds = service_account.Credentials.from_service_account_info(info)
        return texttospeech.TextToSpeechClient(credentials=creds)
    return texttospeech.TextToSpeechClient()

def synthesize_edge_tts(texto: str) -> bytes:
    """
    Mantiene el MISMO nombre para que tu app no truene.
    Genera voz masculina/neutral con Google TTS.
    Devuelve audio MP3.
    """

    if not texto or texto.strip() == "":
        texto = "No se recibió texto para convertir a voz."

    client = _get_client()

    input_text = texttospeech.SynthesisInput(text=texto)

    # 🟦 Voz masculina/neutral es-MX Neural
    voice = texttospeech.VoiceSelectionParams(
        language_code="es-MX",
        name="es-MX-Neural2-D"
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config
    )

    return response.audio_content