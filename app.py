"""
app.py
------
Flask API backend for the Multilingual Localization Engine.
Handles Text, Audio, and PDF inputs -> NLLB Translation -> Gemini Refinement -> TTS.
"""

import os
import uuid
import PyPDF2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import local ML and LLM modules
from translation import translate_text, detect_language, LANG_MAP
from STT import speech_to_text
from TTS import text_to_speech
from refinement import refine_translation

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "api_uploads"
OUTPUT_DIR = "api_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper map for Gemini to know the full language name
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada"
}

@app.route('/api/process', methods=['POST'])
def process_input():
    target_lang = request.form.get('target_language')
    text_input = request.form.get('text')
    uploaded_file = request.files.get('file')
    domain = request.form.get('domain', 'General Conversation') # Default domain

    if not target_lang or target_lang not in LANG_MAP:
        return jsonify({"error": f"Valid 'target_language' required. Options: {list(LANG_MAP.keys())}"}), 400

    if not text_input and not uploaded_file:
        return jsonify({"error": "Must provide either 'text' or a 'file'."}), 400

    source_text = ""

    try:
        # 1. EXTRACT TEXT
        if uploaded_file:
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(UPLOAD_DIR, filename)
            uploaded_file.save(filepath)

            if filename.lower().endswith('.pdf'):
                with open(filepath, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            source_text += extracted + "\n"
            else: # Audio file
                stt_result = speech_to_text(filepath)
                source_text = stt_result.get('text', '')

            if os.path.exists(filepath):
                os.remove(filepath)

        elif text_input:
            source_text = text_input.strip()

        if not source_text.strip():
            return jsonify({"error": "No recognizable text found in the input."}), 400

        # 2. LOCAL ML TRANSLATION (The Draft)
        src_lang = detect_language(source_text)
        draft_translation = translate_text(source_text, tgt_lang_code=target_lang, src_lang_code=src_lang)

        # 3. LLM REFINEMENT (The Polish)
        target_lang_name = LANGUAGE_NAMES.get(target_lang, "English")
        refined_translation = refine_translation(
            original_text=source_text.strip(),
            draft_translation=draft_translation,
            target_language=target_lang_name,
            domain=domain
        )

        # 4. TEXT-TO-SPEECH (Using the Refined Text)
        audio_filename = f"output_{uuid.uuid4().hex[:8]}.wav"
        audio_filepath = os.path.join(OUTPUT_DIR, audio_filename)
        
        text_to_speech(text=refined_translation, lang_code=target_lang, output_path=audio_filepath, play_audio=False)

        # 5. RETURN COMBINED RESPONSE
        return jsonify({
            "source_language": src_lang,
            "target_language": target_lang,
            "domain_used": domain,
            "original_text": source_text.strip(),
            "draft_translation": draft_translation,   # Sending this back so you can compare!
            "refined_translation": refined_translation, # The final LLM output
            "audio_url": f"http://127.0.0.1:5000/api/audio/{audio_filename}"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/audio/<filename>', methods=['GET'])
def get_audio(filename):
    return send_from_directory(OUTPUT_DIR, filename, mimetype="audio/wav")

if __name__ == '__main__':
    print("🚀 Starting Multilingual Localization Engine...")
    app.run(host='0.0.0.0', port=5000, debug=True)