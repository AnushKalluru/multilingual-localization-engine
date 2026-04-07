"""
TTS.py
------
Text-to-Speech (TTS) conversion using OpenAI's Whisper-compatible pipeline.

Since Whisper is a speech recognition model (STT only), TTS uses the
'whisper' ecosystem companion: we use the `whisper` package for consistency
in the project but route TTS through the HuggingFace SpeechT5 pipeline,
which is the closest Whisper-family equivalent for synthesis.

For a truly "Whisper-based" TTS, this module uses:
    openai/whisper  →  transcription  (STT.py)
    microsoft/speecht5_tts  →  synthesis  (TTS.py)
        with speaker embeddings from:
    microsoft/speecht5_hifigan  (vocoder)
    Matthijs/cmu-arctic-xvectors  (speaker embeddings)

Supported languages:
    Whisper's TTS counterpart (SpeechT5) is primarily English.
    For multilingual TTS, the module falls back to Facebook MMS-TTS
    for non-English languages (hi, te, ta, kn).

Output:
    - A .wav file saved to disk
    - Optional live playback
"""

import os
import torch
import numpy as np
import soundfile as sf
from datasets import load_dataset
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    VitsModel,
    AutoTokenizer,
)

# ============================================
# DEVICE
# ============================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# ENGLISH TTS — SpeechT5 (Whisper ecosystem)
# ============================================
SPEECHT5_MODEL   = "microsoft/speecht5_tts"
SPEECHT5_VOCODER = "microsoft/speecht5_hifigan"
SPEAKER_DATASET  = "regisss/cmu-arctic-xvectors"

_speecht5_cache = {}   # loaded lazily

def _load_speecht5():
    if "model" in _speecht5_cache:
        return _speecht5_cache["model"], _speecht5_cache["processor"], _speecht5_cache["vocoder"], _speecht5_cache["embeddings"]

    print("🔄 Loading SpeechT5 TTS model (English)...")
    processor = SpeechT5Processor.from_pretrained(SPEECHT5_MODEL)
    model     = SpeechT5ForTextToSpeech.from_pretrained(SPEECHT5_MODEL).to(device)
    vocoder   = SpeechT5HifiGan.from_pretrained(SPEECHT5_VOCODER).to(device)

    print("🔄 Loading speaker embeddings...")
    embeddings_dataset = load_dataset(SPEAKER_DATASET, split="validation")
    # Use speaker index 7306 — a neutral, clear English voice
    speaker_embeddings = torch.tensor(
        embeddings_dataset[7306]["xvector"]
    ).unsqueeze(0).to(device)

    _speecht5_cache.update({
        "model": model, "processor": processor,
        "vocoder": vocoder, "embeddings": speaker_embeddings,
    })
    print("✅ SpeechT5 loaded!\n")
    return model, processor, vocoder, speaker_embeddings


# ============================================
# MULTILINGUAL TTS — Facebook MMS-TTS fallback
# ============================================
MMS_MODEL_MAP = {
    "hi": "facebook/mms-tts-hin",
    "te": "facebook/mms-tts-tel",
    "ta": "facebook/mms-tts-tam",
    "kn": "facebook/mms-tts-kan",
}

_mms_cache = {}  # { lang_code: (model, tokenizer) }

def _load_mms(lang_code: str):
    if lang_code in _mms_cache:
        return _mms_cache[lang_code]

    model_name = MMS_MODEL_MAP[lang_code]
    print(f"🔄 Loading MMS-TTS model for '{lang_code}': {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = VitsModel.from_pretrained(model_name).to(device)
    model.eval()
    _mms_cache[lang_code] = (model, tokenizer)
    print(f"✅ MMS-TTS '{lang_code}' loaded!\n")
    return model, tokenizer


# ============================================
# CORE TTS FUNCTION
# ============================================
def text_to_speech(
    text: str,
    lang_code: str = "en",
    output_path: str = "output.wav",
    play_audio: bool = False,
) -> str:
    """
    Convert text to speech and save as a .wav file.

    Args:
        text:        The input text to synthesize.
        lang_code:   ISO 639-1 language code ('en', 'hi', 'te', 'ta', 'kn').
        output_path: Destination file path for the output .wav file.
        play_audio:  If True, attempt to play back the audio (requires sounddevice).

    Returns:
        Absolute path to the saved .wav file.

    Raises:
        ValueError: If the language code is not supported.
    """
    supported = {"en"} | set(MMS_MODEL_MAP.keys())
    if lang_code not in supported:
        raise ValueError(f"Unsupported language: '{lang_code}'. Supported: {sorted(supported)}")

    print(f"🗣️  Synthesizing [{lang_code}]: \"{text[:70]}{'...' if len(text) > 70 else ''}\"")

    if lang_code == "en":
        waveform, sample_rate = _synthesize_speecht5(text)
    else:
        waveform, sample_rate = _synthesize_mms(text, lang_code)

    # Normalise
    max_val = np.abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val

    sf.write(output_path, waveform, sample_rate)
    abs_path = os.path.abspath(output_path)
    print(f"💾 Audio saved → {abs_path}")

    if play_audio:
        _play(waveform, sample_rate)

    return abs_path


# ============================================
# SYNTHESIS BACKENDS
# ============================================
def _synthesize_speecht5(text: str):
    """Synthesize English speech using SpeechT5."""
    model, processor, vocoder, speaker_embeddings = _load_speecht5()

    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings,
            vocoder=vocoder,
        )

    waveform = speech.cpu().numpy()
    return waveform, 16_000


def _synthesize_mms(text: str, lang_code: str):
    """Synthesize non-English speech using Facebook MMS-TTS."""
    model, tokenizer = _load_mms(lang_code)

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # MMS-TTS waveform shape: (1, 1, T)
    waveform = outputs.waveform.squeeze().cpu().numpy()
    return waveform, 16_000


# ============================================
# PLAYBACK HELPER
# ============================================
def _play(audio: np.ndarray, sample_rate: int) -> None:
    try:
        import sounddevice as sd
        print("🔊 Playing audio...")
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
    except ImportError:
        print("⚠️  sounddevice not installed. Install with: pip install sounddevice")
    except Exception as e:
        print(f"⚠️  Playback error: {e}")


# ============================================
# EXAMPLE USAGE
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("TTS DEMO — SpeechT5 (en) + MMS-TTS (others)")
    print("=" * 50)

    demos = [
        ("Hello! This is a text to speech demo using SpeechT5.", "en", "demo_en.wav"),
        ("नमस्ते! यह एक वाक् संश्लेषण डेमो है।",               "hi", "demo_hi.wav"),
        ("హలో! ఇది ఒక వాక్ సంశ్లేషణ డెమో.",                   "te", "demo_te.wav"),
        ("வணக்கம்! இது ஒரு உரை-க்கு-பேச்சு டெமோ.",             "ta", "demo_ta.wav"),
        ("ಹಲೋ! ಇದು ಒಂದು ಧ್ವನಿ ಸಂಶ್ಲೇಷಣೆ ಡೆಮೊ.",               "kn", "demo_kn.wav"),
    ]

    for text, lang, fname in demos:
        saved = text_to_speech(text, lang_code=lang, output_path=fname, play_audio=False)
        print(f"  ✔ [{lang}] → {saved}\n")