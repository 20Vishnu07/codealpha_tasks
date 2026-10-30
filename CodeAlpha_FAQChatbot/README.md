# CodeAlpha_FAQChatbot
FAQ chatbot (CodeAlpha AI internship, Task 2).

Pipeline: lowercase and clean, NLTK tokenize, stop-word removal, lemmatize, TF-IDF (1-2 grams), cosine similarity against each FAQ. Below the 0.2 threshold the bot says it has no match instead of guessing.

```
pip install -r requirements.txt
python chatbot.py     # terminal
python app.py         # web UI at http://127.0.0.1:5000
```
Edit `faqs.json` to change the topic.
