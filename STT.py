"""
STT.py
------
Speech-to-Text (STT) conversion using OpenAI's Whisper model.

Whisper supports multilingual transcription and automatic language detection
out of the box, making it ideal for a multilingual translation pipeline.

Supported input:
    - Local .wav / .mp3 / .m4a / .flac file path
    - Raw numpy audio array + sample rate

Models (trade-off between speed and accuracy):
    tiny, base, small, medium, large, large-v2, large-v3
"""

import os
import numpy as np
import soundfile as sf
import torch
import whisper

# ============================================
# MODEL CONFIG
# ============================================
# Change to "medium" or "large-v3" for better multilingual accuracy
WHISPER_MODEL_SIZE = "base"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔄 Loading Whisper '{WHISPER_MODEL_SIZE}' model...")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
print(f"✅ Whisper model loaded on {device.upper()}!\n")

TARGET_SAMPLE_RATE = 16_000  # Whisper expects 16 kHz mono audio


# ============================================
# HELPERS
# ============================================
def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo or multi-channel audio to mono."""
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def _resample_if_needed(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample audio to 16 kHz if necessary."""
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio
    try:
        import librosa
        print(f"🔁 Resampling from {sample_rate} Hz → {TARGET_SAMPLE_RATE} Hz...")
        return librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    except ImportError:
        raise ImportError(
            "librosa is required for resampling non-16kHz audio. "
            "Install it with: pip install librosa"
        )


# ============================================
# CORE STT FUNCTION
# ============================================
def speech_to_text(
    audio_input,
    sample_rate: int = None,
    language: str = None,
    task: str = "transcribe",
) -> dict:
    """
    Transcribe speech from audio to text using Whisper.

    Args:
        audio_input:  File path (str) to an audio file (.wav, .mp3, etc.)
                      OR a numpy array of audio samples.
        sample_rate:  Required only when audio_input is a numpy array.
        language:     Optional ISO 639-1 code to force a language (e.g. 'hi', 'te').
                      If None, Whisper auto-detects the language.
        task:         'transcribe' (keep original language) or
                      'translate' (Whisper translates to English internally).

    Returns:
        dict with keys:
            - 'text'      : transcribed string
            - 'language'  : detected/used language code
            - 'segments'  : list of timed segment dicts from Whisper

    Raises:
        FileNotFoundError: If the given file path does not exist.
        ValueError:        If audio_input type is invalid or sample_rate missing.
    """
    # --- Load / prepare audio ---
    if isinstance(audio_input, str):
        if not os.path.isfile(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        print(f"📂 Loading audio file: {audio_input}")
        audio_path = audio_input

    elif isinstance(audio_input, np.ndarray):
        if sample_rate is None:
            raise ValueError("sample_rate must be provided when passing a numpy array.")

        audio = audio_input.astype(np.float32)
        audio = _to_mono(audio)
        audio = _resample_if_needed(audio, sample_rate)

        tmp_path = "_stt_temp.wav"
        sf.write(tmp_path, audio, TARGET_SAMPLE_RATE)
        audio_path = tmp_path

    else:
        raise ValueError("audio_input must be a file path (str) or a numpy array.")

    # --- Transcribe ---
    options = {
        "task": task,
        "fp16": device == "cuda",
    }
    if language:
        options["language"] = language

    print(f"🎙️  Transcribing... (task={task}, language={'auto' if not language else language})")
    result = whisper_model.transcribe(audio_path, **options)

    # Clean up temp file if created
    if isinstance(audio_input, np.ndarray) and os.path.exists("_stt_temp.wav"):
        os.remove("_stt_temp.wav")

    return {
        "text": result["text"].strip(),
        "language": result.get("language", language or "unknown"),
        "segments": result.get("segments", []),
    }

#============================================
# EXAMPLE USAGE
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("STT DEMO — Whisper")
    print("=" * 50)

    # Point this to the actual audio file you generated with TTS.py!
    REAL_AUDIO_FILE = "demo_ta.wav" 

    # Run the transcription
    result = speech_to_text(REAL_AUDIO_FILE, language=None)  # auto-detect language

    print(f"📝 Transcription : {result['text']!r}")
    print(f"🌐 Language      : {result['language']}")