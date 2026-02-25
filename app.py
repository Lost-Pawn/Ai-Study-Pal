import os
import random
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request, send_file
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

app = Flask(__name__)

GLOVE_PATH = "data/glove.6B.50d.txt"
glove_embeddings = {}
keras_model = None
keras_tokenizer = None
tfidf_vectorizer = None

tips_template = {
    "dsa": [
        "Implement each data structure in code to understand its behavior.",
        "Solve at least one algorithm problem daily to strengthen problem-solving skills.",
        "Trace your code with examples to visualize how data moves through structures.",
        "Focus on time and space complexity analysis for every algorithm you learn.",
        "Practice coding common problems like sorting, searching, and recursion."
    ],
    "programming": [
        "Build small projects applying concepts like loops, functions, and classes.",
        "Read and debug existing code to understand different programming styles.",
        "Practice writing clean and modular code with proper comments.",
        "Learn to use version control tools like Git while coding projects.",
        "Implement OOP concepts by creating real-world applications using classes and inheritance."
    ],
    "cloud engineering": [
        "Deploy a simple application on a cloud platform to understand the workflow.",
        "Practice setting up and managing virtual machines and storage resources.",
        "Understand key cloud concepts like IaaS, PaaS, and SaaS with examples.",
        "Explore monitoring and logging tools provided by cloud platforms.",
        "Learn to design scalable architectures using cloud services for practice projects."
    ]
}

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

resources_links = {
    "programming": [
        "https://www.geeksforgeeks.org/",
        "https://www.w3schools.com/",
        "https://www.tutorialspoint.com/"
    ],
    "dsa": [
        "https://leetcode.com/",
        "https://www.hackerrank.com/domains/tutorials",
        "https://codeforces.com/"
    ],
    "cloud engineering": [
        "https://aws.amazon.com/training/",
        "https://learn.microsoft.com/en-us/training/",
        "https://cloud.google.com/training"
    ]
}

def load_glove():
    global glove_embeddings
    try:
        with open(GLOVE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                glove_embeddings[parts[0]] = np.array(parts[1:], dtype="float32")
        print("GloVe loaded.")
    except FileNotFoundError:
        print("GloVe file not found. Using empty embeddings.")

def get_text_vector(text, dim=50):
    words = text.lower().split()
    vectors = [glove_embeddings[w] for w in words if w in glove_embeddings]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

def setup_models():
    global keras_model, keras_tokenizer, tfidf_vectorizer
    print("Initializing Models...")
    X_train = ["This is a test sentence.", "Machine learning is fascinating.", "Cloud computing scales well."]
    y_train = np.array([0.8, 0.9, 0.7])
    keras_tokenizer = Tokenizer(oov_token="<OOV>")
    keras_tokenizer.fit_on_texts(X_train)
    vocab_size = len(keras_tokenizer.word_index) + 1
    X_seq = pad_sequences(keras_tokenizer.texts_to_sequences(X_train), maxlen=30, padding="post")

    keras_model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=32, input_length=30),
        GlobalAveragePooling1D(),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    keras_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    keras_model.fit(X_seq, y_train, epochs=1, verbose=0) 
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
    tfidf_vectorizer.fit(X_train)
    print("Models ready.")

def generate_summary(text, max_words=50):
    sentences = sent_tokenize(text)
    if not sentences: return text
    seqs = pad_sequences(keras_tokenizer.texts_to_sequences(sentences), maxlen=30, padding="post")
    scores = keras_model.predict(seqs, verbose=0).flatten()
    ranked = sorted(zip(scores, sentences), reverse=True)
    summary_sentences, word_count = [], 0

    for i, (score, sent) in enumerate(ranked):
        words = sent.split()
        if i == 0 or word_count + len(words) <= max_words:
            summary_sentences.append(sent)
            word_count += len(words)
    summary_sentences.sort(key=lambda x: sentences.index(x))
    return " ".join(summary_sentences)

def generate_quiz(text, num_questions=10):
    sentences = sent_tokenize(text)
    if not sentences: return []
    try:
        dynamic_vectorizer = TfidfVectorizer(stop_words="english")
        vec = dynamic_vectorizer.fit_transform(sentences)
        terms = dynamic_vectorizer.get_feature_names_out()
        scores = vec.sum(axis=0).A1
        sorted_indices = scores.argsort()[::-1]
        ranked_terms = [terms[i] for i in sorted_indices if len(terms[i]) > 3]
        quiz_list = []
        used_sentences = set()
        for term in ranked_terms:
            if len(quiz_list) >= num_questions:
                break
            for i, sent in enumerate(sentences):
                if i in used_sentences:
                    continue
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                if pattern.search(sent) and len(sent.split()) > 4:
                    question = pattern.sub("_____", sent)
                    distractors = [t.title() for t in ranked_terms if t != term]
                    random.shuffle(distractors)
                    options = [term.title()] + distractors[:3]
                    while len(options) < 4:
                        options.append(random.choice(["Function", "Process", "System", "None of the Above"]))
                    random.shuffle(options)
                    quiz_list.append({
                        "question": question,
                        "options": options,
                        "answer": term.title()
                    })
                    used_sentences.add(i)
                    break
        return quiz_list

    except Exception as e:
        print(f"Quiz generation bypassed: {e}")    
        return []

def generate_study_plan(subject, hours, start_time="18:00"):
    try:
        hrs = int(hours)
    except ValueError:
        hrs = 2
    subject_topics = {
        "dsa": ["Arrays", "Linked List", "Stack", "Queue", "Trees", "Graphs"],
        "programming": ["Variables", "Loops", "Functions", "OOP", "File Handling"],
        "cloud": ["AWS Basics", "EC2", "S3", "IAM", "VPC"]
    }

    topics = subject_topics.get(subject.lower(), [f"{subject.title()} Basics"])

    total_minutes = hrs * 60
    session_duration = 45
    break_duration = 10

    current_time = datetime.strptime(start_time, "%H:%M")
    minutes_used = 0
    session_count = 0
    plan = []

    study_types = ["Theory", "Practice", "Revision"]

    while minutes_used + session_duration <= total_minutes:
        topic = topics[session_count % len(topics)]
        study_type = study_types[session_count % len(study_types)]

        start = current_time
        end = current_time + timedelta(minutes=session_duration)

        plan.append({
            "Session": f"Session {session_count+1}",
            "Start Time": start.strftime("%H:%M"),
            "End Time": end.strftime("%H:%M"),
            "Type": study_type,
            "Focus": topic,
            "Duration": f"{session_duration} mins"
        })

        current_time = end + timedelta(minutes=break_duration)
        minutes_used += session_duration + break_duration
        session_count += 1

    os.makedirs("data", exist_ok=True)
    pd.DataFrame(plan).to_csv("data/study_schedule.csv", index=False)

    return plan

def get_feedback_and_tips(subject, text):
    sub_lower = subject.lower()
    tokens = word_tokenize(text.lower())
    filtered = [w for w in tokens if w.isalpha() and w not in stop_words]
    keywords = [w for w, _ in FreqDist(filtered).most_common(3)]
    tips = tips_template.get(sub_lower, ["Review daily.", "Take notes.", "Practice questions."])
    phrase = subject_phrases.get(sub_lower, subject)
    svec = get_text_vector(f"{subject} {phrase}")
    fvecs = [get_text_vector(t["text"]) for t in feedback_templates]
    sims = [np.dot(svec, fv) / (np.linalg.norm(svec) * np.linalg.norm(fv) + 1e-8) if not np.all(fv==0) else 0 for fv in fvecs]
    feedback = feedback_templates[int(np.argmax(sims))]["text"] if any(sims) else feedback_templates[0]["text"]
    resources = resources_links.get(sub_lower, ["Google Scholar", "Wikipedia"])
    return tips, keywords, feedback, resources

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    subject = request.form.get("subject", "General")
    hours = request.form.get("hours", "2")
    text = request.form.get("text", "")
    summary = generate_summary(text)
    quiz = generate_quiz(text)
    plan = generate_study_plan(subject, hours)
    tips, keywords, feedback, resources = get_feedback_and_tips(subject, text)
    return render_template("results.html", 
                           subject=subject, summary=summary, quiz=quiz, 
                           plan=plan, tips=tips, keywords=keywords, 
                           feedback=feedback, resources=resources)

@app.route("/download_schedule")
def download():
    return send_file("data/study_schedule.csv", as_attachment=True)

if __name__ == "__main__":
    load_glove()
    setup_models()
    app.run(debug=True)