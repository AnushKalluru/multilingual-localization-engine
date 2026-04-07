import os
import torch
import numpy as np
import soundfile as sf
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, VitsModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

SPEECHT5_MODEL   = "microsoft/speecht5_tts"
SPEECHT5_VOCODER = "microsoft/speecht5_hifigan"
SPEAKER_DATASET  = "regisss/cmu-arctic-xvectors"
_speecht5_cache = {}

MMS_MODEL_MAP = {
    "hi": "facebook/mms-tts-hin",
    "te": "facebook/mms-tts-tel",
    "ta": "facebook/mms-tts-tam",
    "kn": "facebook/mms-tts-kan",
}
_mms_cache = {}

def _load_speecht5():
    if "model" in _speecht5_cache:
        return _speecht5_cache["model"], _speecht5_cache["processor"], _speecht5_cache["vocoder"], _speecht5_cache["embeddings"]

    print("🔄 Loading SpeechT5 TTS model...")
    processor = SpeechT5Processor.from_pretrained(SPEECHT5_MODEL)
    model = SpeechT5ForTextToSpeech.from_pretrained(SPEECHT5_MODEL).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(SPEECHT5_VOCODER).to(device)
    embeddings_dataset = load_dataset(SPEAKER_DATASET, split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

    _speecht5_cache.update({"model": model, "processor": processor, "vocoder": vocoder, "embeddings": speaker_embeddings})
    return model, processor, vocoder, speaker_embeddings

def _load_mms(lang_code: str):
    if lang_code in _mms_cache: return _mms_cache[lang_code]
    model_name = MMS_MODEL_MAP[lang_code]
    print(f"🔄 Loading MMS-TTS for '{lang_code}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name).to(device)
    model.eval()
    _mms_cache[lang_code] = (model, tokenizer)
    return model, tokenizer

def text_to_speech(text: str, lang_code: str = "en", output_path: str = "output.wav", play_audio: bool = False) -> str:
    supported = {"en"} | set(MMS_MODEL_MAP.keys())
    if lang_code not in supported: raise ValueError(f"Unsupported language: '{lang_code}'")

    if lang_code == "en":
        model, processor, vocoder, speaker_embeddings = _load_speecht5()
        inputs = processor(text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        waveform = speech.cpu().numpy()
    else:
        model, tokenizer = _load_mms(lang_code)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        waveform = outputs.waveform.squeeze().cpu().numpy()

    max_val = np.abs(waveform).max()
    if max_val > 0: waveform = waveform / max_val

    sf.write(output_path, waveform, 16000)
    return os.path.abspath(output_path)