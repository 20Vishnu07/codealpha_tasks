"""Translation tool: Flask UI + Google Translate via deep-translator (no API key).
For the official paid API, swap translate_text() for google-cloud-translate."""
from flask import Flask, jsonify, render_template, request
from deep_translator import GoogleTranslator

app = Flask(__name__)
MAX_CHARS = 4500  # Google web endpoint limit is ~5000

def languages():
    # {"english": "en", ...}
    return GoogleTranslator().get_supported_languages(as_dict=True)

def translate_text(text, source, target):
    return GoogleTranslator(source=source, target=target).translate(text)

@app.get("/")
def index():
    try:
        langs = languages()
    except Exception:
        langs = {"english": "en", "tamil": "ta", "hindi": "hi", "french": "fr", "spanish": "es", "german": "de"}
    return render_template("index.html", langs=sorted(langs.items()))

@app.post("/translate")
def translate():
    d = request.get_json(silent=True) or {}
    text, src, tgt = (d.get("text") or "").strip(), d.get("source", "auto"), d.get("target", "en")
    if not text:
        return jsonify(error="Enter some text to translate."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Text is too long. Limit is {MAX_CHARS} characters."), 400
    if src == tgt:
        return jsonify(translation=text)
    try:
        return jsonify(translation=translate_text(text, src, tgt))
    except Exception as e:
        return jsonify(error=f"Translation failed ({type(e).__name__}). Check your internet connection and language choice."), 502

if __name__ == "__main__":
    app.run(debug=True)
