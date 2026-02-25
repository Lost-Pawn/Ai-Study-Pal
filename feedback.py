import os
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

P, S, C = "paragraph", "subject", "category"

df = pd.read_csv("data/generated_mcqs.csv")
df[P] = df[P].fillna("").astype(str)
paragraphs = df[P].tolist()
subjects = df[S].fillna("").astype(str).tolist()
categories = df.get(C, pd.Series([""] * len(df))).astype(str).tolist()

def split_sentences(text):
    return sent_tokenize(text)

def build_training_data(paragraphs):
    X, y = [], []
    for text in paragraphs:
        sentences = split_sentences(text)
        total = len(sentences)
        if total == 0: continue
        for i, sent in enumerate(sentences):
            X.append(sent)
            y.append(round(((1 - (i / total)) + min(len(sent.split()) / 20, 1.0)) / 2, 2))
    return X, y

X_data, y_data = build_training_data(paragraphs)

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X_data)
vocab_size = len(tokenizer.word_index) + 1

X_seq = pad_sequences(tokenizer.texts_to_sequences(X_data), maxlen=30, padding="post")
y_arr = np.array(y_data)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=30),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(X_seq, y_arr, epochs=20, batch_size=4, verbose=1)

def summarize(text, model, tokenizer, max_len=30, max_words=50):
    sentences = split_sentences(text)
    if not sentences: return ""
    seqs = pad_sequences(tokenizer.texts_to_sequences(sentences), maxlen=max_len, padding="post")
    scores = model.predict(seqs, verbose=1).flatten()
    ranked = sorted(zip(scores, sentences), reverse=True)
    summary_sentences = []
    word_count = 0
    for i, (score, sent) in enumerate(ranked):
        words = sent.split()
        if i == 0 or word_count + len(words) <= max_words:
            summary_sentences.append(sent)
            word_count += len(words)
        if word_count >= max_words: break
    summary_sentences.sort(key=lambda x: sentences.index(x))
    return " ".join(summary_sentences)

def load_glove(path):
    embeddings = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                embeddings[parts[0]] = np.array(parts[1:], dtype="float32")
        return embeddings
    except FileNotFoundError: return embeddings

def get_text_vector(text, embeddings, dim=50):
    words = text.lower().split()
    vectors = [embeddings[w] for w in words if w in embeddings]
    if not vectors: return np.zeros(dim)
    return np.mean(vectors, axis=0)

glove_embeddings = load_glove("data/glove.6B.50d.txt")

feedback_templates = [
    {"text": "Great work! Keep practicing and you will master this topic.", "tag": "positive"},
    {"text": "Good effort! Review the key concepts and try again.", "tag": "review"},
    {"text": "You are making progress! Stay consistent with your study sessions.", "tag": "progress"},
    {"text": "Keep it up! A little practice every day goes a long way.", "tag": "encouragement"},
    {"text": "Well done! Try challenging yourself with harder questions next.", "tag": "challenge"},
    {"text": "Nice job! Make sure to revise before your next session.", "tag": "revise"},
    {"text": "You are on the right track! Focus on the areas you find difficult.", "tag": "focus"},
    {"text": "Good job on this subject! Consistency is the key to success.", "tag": "consistency"},
]

subject_phrases = {
    "programming": "writing programs, debugging code, practicing algorithms, and learning programming concepts",
    "dsa": "analyzing algorithms, implementing data structures, practicing coding problems, and solving computational challenges",
    "cloud engineering": "deploying applications, managing cloud infrastructure, understanding scalable architecture, and exploring cloud services",
}

feedback_vectors = np.array([get_text_vector(t["text"], glove_embeddings) for t in feedback_templates])

def get_feedback(sub, cat, emb, fvecs, ftmps):
    svec = get_text_vector(f"{sub} {subject_phrases.get(cat.lower(), sub)}", emb)
    if np.all(svec == 0): return ftmps[0]["text"]
    sims = []
    for fv in fvecs:
        if np.all(fv == 0): sims.append(0)
        else: sims.append(np.dot(svec, fv) / (np.linalg.norm(svec) * np.linalg.norm(fv) + 1e-8))
    return ftmps[int(np.argmax(sims))]["text"]

summary_rows, feedback_rows = [], []

for p, s, c in zip(paragraphs, subjects, categories):
    if not p.strip(): continue
    summary_rows.append({S: s, P: p, "summary": summarize(p, model, tokenizer)})
    feedback_rows.append({S: s, "feedback": get_feedback(s, c, glove_embeddings, feedback_vectors, feedback_templates)})

pd.DataFrame(summary_rows).to_csv("data/summaries.csv", index=False)
pd.DataFrame(feedback_rows).to_csv("data/feedback.csv", index=False)