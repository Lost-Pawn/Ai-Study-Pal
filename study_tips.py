import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

df = pd.read_csv("data/generated_mcqs.csv")

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

generic_tips = [
    "Review the main concepts of the topic daily to retain knowledge.",
    "Summarize what you implemented or studied in your own words.",
    "Test yourself with coding exercises or scenario-based questions.",
    "Draw diagrams or flowcharts to visualize processes and algorithms.",
    "Take small breaks to stay focused and prevent burnout while learning."
]

def extract_keywords(text, top_n=4):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    filtered = [
        word for word in tokens
        if word.isalpha() and word not in stop_words and len(word) > 3
    ]
    freq = FreqDist(filtered)
    return [word for word, _ in freq.most_common(top_n)]

def generate_tips(category):
    return tips_template.get(category.strip().lower(), generic_tips)[:3]

print("Study Tips Generator: \n")

results = []

for _, row in df.iterrows():
    subject = row["subject"]
    category = row["category"]
    text = row["paragraph"]

    sentences = sent_tokenize(text)
    keywords = extract_keywords(text, top_n=4)
    tips = generate_tips(category)

    print(f"Subject    : {subject.title()}")
    print(f"Category   : {category.title()}")
    print(f"Sentences  : {len(sentences)}")
    print(f"Keywords   : {', '.join(keywords)}")
    print("Study Tips :")
    for i, tip in enumerate(tips, 1):
        print(f"  {i}. {tip}")

    results.append({
        "subject": subject,
        "category": category,
        "paragraph": text,
        "sentences_count": len(sentences),
        "top_keywords": ", ".join(keywords),
        "tip_1": tips[0],
        "tip_2": tips[1],
        "tip_3": tips[2],
    })

df_tips = pd.DataFrame(results)
df_tips.to_csv("data/study_tips.csv", index=False)

print("Done\n")