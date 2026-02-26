import pandas as pd
import random
import nltk
import re

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt_tab", quiet=True)

df = pd.read_csv("data/clean_data.csv")
df = df.dropna(subset=["text", "subject", "difficulty"])

df["text"] = df["text"].astype(str)
df = df[df["text"].str.len() > 100]

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 3),
    max_features=8000,
    min_df=1
)

vectorizer.fit(df["text"]) 
terms = vectorizer.get_feature_names_out()

def extract_answer(text): # extracts the answer from the paragraph
    vec = vectorizer.transform([text])
    scores = vec.toarray()[0]
    sorted_indices = scores.argsort()[::-1]

    for idx in sorted_indices:
        candidate = terms[idx]
        if len(candidate) > 3 and not candidate.isdigit():
            return candidate

    return None

def create_question(text, answer): 
    sentences = sent_tokenize(text)
    for sentence in sentences:
        if (
            answer.lower() in sentence.lower()
            and len(sentence.split()) > 6
        ):
            pattern = re.compile(re.escape(answer), re.IGNORECASE) # puts a blank in the place with answer to make quiz
            return pattern.sub("_____", sentence)
    return None

def get_distractors(answer, all_terms, n=3): # makes 3 distractors 
    distractors = [
        t for t in all_terms
        if t.lower() != answer.lower()
        and len(t) > 3
        and not t.isdigit()
        and t[0].isalpha()
    ]
    if len(distractors) < n:
        return None
    return random.sample(distractors, n)

quiz_data = []
seen_answers = {}

for row in df.itertuples():
    paragraph = row.text
    subject = row.subject
    answer = extract_answer(paragraph)

    if subject not in seen_answers:
        seen_answers[subject] = set()

    if not answer or answer in seen_answers[subject]:
        continue

    question = create_question(paragraph, answer)
    if not question:
        continue

    distractors = get_distractors(answer, terms.tolist())
    if not distractors:
        continue

    options = distractors + [answer]
    random.shuffle(options) # and suffles them everytime

    quiz_data.append({
        "subject": subject,
        "paragraph": paragraph,
        "question": question,
        "difficulty": row.difficulty,
        "option_A": options[0],
        "option_B": options[1],
        "option_C": options[2],
        "option_D": options[3],
        "correct_answer": answer
    })

    seen_answers[subject].add(answer)

final_quiz_df = pd.DataFrame(quiz_data)
final_quiz_df.to_csv("data/generated_mcqs.csv", index=False)

print("Generated:", len(final_quiz_df))
