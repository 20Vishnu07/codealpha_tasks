"""FAQ chatbot: NLTK preprocessing + TF-IDF + cosine similarity."""
import json, re, sys
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

_STOP = set(stopwords.words("english")) - {"not", "no"}
_LEM = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = [_LEM.lemmatize(t) for t in word_tokenize(text) if t not in _STOP]
    return " ".join(tokens)

class FAQBot:
    def __init__(self, path="faqs.json", threshold=0.2):
        self.faqs = json.loads(Path(path).read_text(encoding="utf-8"))
        self.threshold = threshold
        # Index question + answer so keyword-only queries still match
        docs = [preprocess(f["q"] + " " + f["q"] + " " + f["a"]) for f in self.faqs]
        self.vec = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform(docs)

    def ask(self, query: str):
        q = preprocess(query)
        if not q.strip():
            return {"answer": "Please type a question.", "score": 0.0, "matched": None}
        scores = cosine_similarity(self.vec.transform([q]), self.matrix)[0]
        i = int(scores.argmax())
        if scores[i] < self.threshold:
            return {"answer": "I could not find a close match. Try rephrasing, or email services@codealpha.tech.",
                    "score": float(scores[i]), "matched": None}
        return {"answer": self.faqs[i]["a"], "score": float(scores[i]), "matched": self.faqs[i]["q"]}

if __name__ == "__main__":
    bot = FAQBot(Path(__file__).with_name("faqs.json"))
    print("FAQ bot (type 'quit' to exit)")
    while True:
        try:
            msg = input("you> ").strip()
        except EOFError:
            break
        if msg.lower() in {"quit", "exit"}:
            break
        r = bot.ask(msg)
        print(f"bot> {r['answer']}  [score={r['score']:.2f}]")
