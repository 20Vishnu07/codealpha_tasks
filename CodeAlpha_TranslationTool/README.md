# CodeAlpha_TranslationTool
Language translation web app (CodeAlpha AI internship, Task 1).

- Text box, source (or auto-detect) and target language selectors, swap button
- Translation through Google Translate using the `deep-translator` package (no API key needed)
- Copy button and text-to-speech (browser Web Speech API; voice availability depends on your OS/browser, Tamil may need a language pack)

```
pip install -r requirements.txt
python app.py     # http://127.0.0.1:5000
```
Needs internet access. To use the official Google Cloud Translation API, replace `translate_text()` in `app.py`.
