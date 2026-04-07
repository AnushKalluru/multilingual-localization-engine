import os
import numpy as np
import soundfile as sf
import torch
import whisper

WHISPER_MODEL_SIZE = "base"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔄 Loading Whisper '{WHISPER_MODEL_SIZE}' STT model...")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)

TARGET_SAMPLE_RATE = 16_000

def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1: return audio.mean(axis=1)
    return audio

def _resample_if_needed(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SAMPLE_RATE: return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    except ImportError:
        raise ImportError("librosa is required for resampling.")

def speech_to_text(audio_input, sample_rate: int = None, language: str = None, task: str = "transcribe") -> dict:
    if isinstance(audio_input, str):
        if not os.path.isfile(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        audio_path = audio_input

    elif isinstance(audio_input, np.ndarray):
        if sample_rate is None: raise ValueError("sample_rate must be provided.")
        audio = _resample_if_needed(_to_mono(audio_input.astype(np.float32)), sample_rate)
        tmp_path = "_stt_temp.wav"
        sf.write(tmp_path, audio, TARGET_SAMPLE_RATE)
        audio_path = tmp_path
    else:
        raise ValueError("audio_input must be a file path or numpy array.")

    options = {"task": task, "fp16": device == "cuda"}
    if language: options["language"] = language

    result = whisper_model.transcribe(audio_path, **options)

    if isinstance(audio_input, np.ndarray) and os.path.exists("_stt_temp.wav"):
        os.remove("_stt_temp.wav")

    return {
        "text": result["text"].strip(),
        "language": result.get("language", language or "unknown"),
        "segments": result.get("segments", []),
    }