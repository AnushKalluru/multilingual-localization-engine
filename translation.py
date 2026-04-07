"""
translation.py
--------------
Detects the language of input text and translates it to a desired target language
using Facebook's NLLB-200 distilled model via HuggingFace Transformers.

Supported languages:
    en  → English   (eng_Latn)
    hi  → Hindi     (hin_Deva)
    te  → Telugu    (tel_Telu)
    ta  → Tamil     (tam_Taml)
    kn  → Kannada   (kan_Knda)
"""

from langdetect import detect, LangDetectException
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# ============================================
# LANGUAGE MAP (NLLB Supported Codes)
# ============================================
LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "kn": "kan_Knda",
}

REVERSE_LANG_MAP = {v: k for k, v in LANG_MAP.items()}

# ============================================
# LOAD MODEL (NLLB-200 Distilled 600M)
# ============================================
MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("🔄 Loading translation model... (first run may take a while)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"✅ Translation model loaded on {device.upper()}!\n")


# ============================================
# LANGUAGE DETECTION
# ============================================
def detect_language(text: str) -> str:
    """
    Detect the ISO 639-1 language code of the given text.

    Args:
        text: Input string to detect.

    Returns:
        ISO 639-1 code (e.g. 'en', 'hi'). Falls back to 'en' on failure.
    """
    try:
        detected = detect(text)
        if detected not in LANG_MAP:
            print(f"⚠️  Detected language '{detected}' is not in the supported map. Defaulting to 'en'.")
            return "en"
        return detected
    except LangDetectException as e:
        print(f"⚠️  Language detection failed: {e}. Defaulting to 'en'.")
        return "en"


# ============================================
# TRANSLATION
# ============================================
def translate_text(text: str, tgt_lang_code: str, src_lang_code: str = None) -> str:
    """
    Translate text from a source language to a target language.

    Args:
        text:           The input text to translate.
        tgt_lang_code:  Target ISO 639-1 code (e.g. 'hi', 'ta').
        src_lang_code:  Source ISO 639-1 code. Auto-detected if not provided.

    Returns:
        Translated string.

    Raises:
        ValueError: If source or target language codes are unsupported.
    """
    if src_lang_code is None:
        src_lang_code = detect_language(text)
        print(f"🔍 Auto-detected source language: {src_lang_code}")

    if src_lang_code not in LANG_MAP:
        raise ValueError(f"Unsupported source language: '{src_lang_code}'. Supported: {list(LANG_MAP.keys())}")
    if tgt_lang_code not in LANG_MAP:
        raise ValueError(f"Unsupported target language: '{tgt_lang_code}'. Supported: {list(LANG_MAP.keys())}")

    src_nllb = LANG_MAP[src_lang_code]
    tgt_nllb = LANG_MAP[tgt_lang_code]

    if src_nllb == tgt_nllb:
        print("ℹ️  Source and target languages are the same. Returning original text.")
        return text

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


# ============================================
# EXAMPLE USAGE
# ============================================
if __name__ == "__main__":
    sample_text = "Hello, world! I am testing the translation engine."

    print("=" * 50)
    print("TRANSLATION DEMO")
    print("=" * 50)
    print(f"Source text : {sample_text}")

    detected = detect_language(sample_text)
    print(f"Detected    : {detected} ({LANG_MAP.get(detected, 'unknown')})")

    targets = [code for code in LANG_MAP if code != detected]
    for tgt in targets:
        result = translate_text(sample_text, tgt_lang_code=tgt, src_lang_code=detected)
        print(f"→ {tgt.upper()} ({LANG_MAP[tgt]}): {result}")