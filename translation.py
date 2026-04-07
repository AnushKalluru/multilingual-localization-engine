from langdetect import detect, LangDetectException
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "kn": "kan_Knda",
}

MODEL_NAME = "facebook/nllb-200-distilled-600M"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔄 Loading NLLB Translation model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

def detect_language(text: str) -> str:
    try:
        detected = detect(text)
        return detected if detected in LANG_MAP else "en"
    except LangDetectException:
        return "en"

def translate_text(text: str, tgt_lang_code: str, src_lang_code: str = None) -> str:
    if src_lang_code is None:
        src_lang_code = detect_language(text)

    if src_lang_code not in LANG_MAP or tgt_lang_code not in LANG_MAP:
        raise ValueError("Unsupported source or target language.")

    src_nllb = LANG_MAP[src_lang_code]
    tgt_nllb = LANG_MAP[tgt_lang_code]

    if src_nllb == tgt_nllb: return text

    tokenizer.src_lang = src_nllb
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_nllb)
    
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512,
            num_beams=4,
            early_stopping=True,
        )

    output = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return output[0]