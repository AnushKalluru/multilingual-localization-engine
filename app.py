"""
app.py
------
Flask API backend for the Multilingual Localization Engine.
Handles Text, Audio, and PDF inputs to generate translated text and audio.
"""

import os
import uuid
import PyPDF2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import local ML modules
app = Flask(__name__)
CORS(app)

# Directories for temporary files
UPLOAD_DIR = "api_uploads"
OUTPUT_DIR = "api_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# MAIN PROCESSING ROUTE
# ==========================================
@app.route('/api/process', methods=['POST'])
def process_input():
    """
    Accepts: target_language (required)
             text (optional string)
             file (optional PDF or Audio file)
    Returns: JSON with original text, translated text, and an audio download URL.
    """
    target_lang = request.form.get('target_language')
    text_input = request.form.get('text')
    uploaded_file = request.files.get('file')

    if not target_lang or target_lang not in LANG_MAP:
        return jsonify({"error": f"Valid 'target_language' required. Options: {list(LANG_MAP.keys())}"}), 400

    if not text_input and not uploaded_file:
        return jsonify({"error": "Must provide either 'text' or a 'file'."}), 400

    source_text = ""

    try:
        # 1. HANDLE INPUT EXTRACTION
        if uploaded_file:
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(UPLOAD_DIR, filename)
            uploaded_file.save(filepath)

            # Route A: PDF File
            if filename.lower().endswith('.pdf'):
                with open(filepath, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            source_text += extracted + "\n"
            
            # Route B: Audio File (Speech-to-Text)
            else:
                stt_result = speech_to_text(filepath)
                source_text = stt_result.get('text', '')

            # Cleanup upload
            if os.path.exists(filepath):
                os.remove(filepath)

        # Route C: Direct Text
        elif text_input:
            source_text = text_input.strip()

        if not source_text.strip():
            return jsonify({"error": "No recognizable text found in the input."}), 400

        # 2. HANDLE TRANSLATION
        src_lang = detect_language(source_text)
        translated_text = translate_text(source_text, tgt_lang_code=target_lang, src_lang_code=src_lang)

        # 3. HANDLE TEXT-TO-SPEECH
        audio_filename = f"output_{uuid.uuid4().hex[:8]}.wav"
        audio_filepath = os.path.join(OUTPUT_DIR, audio_filename)
        
        text_to_speech(text=translated_text, lang_code=target_lang, output_path=audio_filepath, play_audio=False)

        # 4. RETURN RESPONSE
        return jsonify({
            "source_language": src_lang,
            "target_language": target_lang,
            "original_text": source_text.strip(),
            "translated_text": translated_text,
            "audio_url": f"http://127.0.0.1:5000/api/audio/{audio_filename}"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
# AUDIO FILE SERVING ROUTE
# ==========================================
@app.route('/api/audio/<filename>', methods=['GET'])
def get_audio(filename):
    """Serves the generated .wav files to the frontend."""
    return send_from_directory(OUTPUT_DIR, filename, mimetype="audio/wav")


if __name__ == '__main__':
    print("🚀 Starting API Engine...")
    app.run(host='0.0.0.0', port=5000, debug=True)