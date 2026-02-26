import pandas as pd
import wikipedia
import time
import random

topics = [
    # ================= DSA =================
    "Introduction to Data Structures", "Arrays and Strings", "Linked Lists",
    "Stacks and Queues", "Trees and Binary Trees", "Binary Search Trees",
    "Heaps and Priority Queues", "Graphs and Graph Traversals",
    "Hashing and Hash Tables", "Recursion and Backtracking",
    "Sorting Algorithms", "Searching Algorithms", "Dynamic Programming",
    "Greedy Algorithms", "Divide and Conquer Techniques",

    # ================= PROGRAMMING =================
    "Introduction to Programming", "Variables and Data Types", "Operators and Expressions",
    "Control Flow Statements", "Loops: for, while, nested loops", "Functions and Recursion",
    "Object-Oriented Programming Concepts", "Classes and Objects",
    "Inheritance and Polymorphism", "Encapsulation and Abstraction",
    "Exception Handling", "File I/O Operations", "Modules and Packages",
    "Debugging and Testing", "Version Control with Git",

    # ================= CLOUD ENGINEERING =================
    "Introduction to Cloud Engineering", "Cloud Service Models: IaaS, PaaS, SaaS",
    "Public, Private, and Hybrid Clouds", "Virtual Machines and Containers",
    "Storage Services in the Cloud", "Networking in Cloud Platforms",
    "Cloud Security and Identity Management", "Monitoring and Logging",
    "Serverless Computing", "Scaling Applications and Load Balancing",
    "Deploying Applications to AWS/Azure/GCP", "Cloud Architecture Best Practices",
    "Cost Optimization and Resource Management", "Disaster Recovery in Cloud",
    "CI/CD Pipelines and Automation in Cloud",
]

def fetch_wikipedia_summary(topic, sentences=30):
    try:
        summary = wikipedia.summary(topic, sentences=sentences, auto_suggest=False)
        return summary, topic
    except wikipedia.DisambiguationError as e:
        print(f" Disambiguation: '{topic}', trying: '{e.options[0]}'")

        try: #recheck as e.topic
            summary = wikipedia.summary(e.options[0], sentences=sentences, auto_suggest=False)
            return summary, e.options[0]
        except Exception as inner_e:
            print(f" Fallback failed: {inner_e}")

    except wikipedia.PageError:
        pass 
    except Exception as e:
        print(f" Error : {e}")

    try: # last level
        search_results = wikipedia.search(topic, results=3)
        if not search_results:
            print(f" No results for '{topic}'")
            return None, None
        
        for candidate in search_results:
            try:
                summary = wikipedia.summary(candidate, sentences=sentences, auto_suggest=False)
                print(f" Used search result '{candidate}' for '{topic}'")
                return summary, candidate

            except wikipedia.DisambiguationError as e:
                try:
                    summary = wikipedia.summary(e.options[0], sentences=sentences, auto_suggest=False)
                    return summary, e.options[0]
                except:
                    continue
            except:
                continue
        
        print(f" All candidates failed for '{topic}'")
        return None, None

    except Exception as e:
        print(f" Search itself failed for '{topic}': {e}")
        return None, None


def assign_difficulty(text):

    words = text.split()
    if not words:
        return "Medium", 0, 0.0

    sentences = [s.strip() for s in text.split('.') if s.strip()]
    avg_sentence_len = len(words) / max(len(sentences), 1)

    avg_word_len = sum(len(w.strip(".,!?()[]")) for w in words) / len(words)

    score = 0
    if avg_word_len > 5.5:
        score += 2
    elif avg_word_len > 4.8:
        score += 1

    if avg_sentence_len > 25:
        score += 2
    elif avg_sentence_len > 18:
        score += 1

    if score >= 3:
        difficulty = "Hard"
    elif score >= 1:
        difficulty = "Medium"
    else:
        difficulty = "Easy"

    return difficulty, len(words), round(avg_word_len, 2)


data_list = []
wikipedia.set_lang("en") 

print("--- STARTING TO FETCH RAW DATA ---\n")

for i, topic in enumerate(topics, 1):
    print(f"[{i}/{len(topics)}] Fetching: {topic}")

    summary, used_title = fetch_wikipedia_summary(topic)

    difficulty, word_count, avg_word_len = assign_difficulty(summary)
    calculated_hours = max(1, round(word_count / 50))

    data_list.append({
        "subject":             topic,                 
        "text":                summary,
        "difficulty":          difficulty,
        "study_hours_needed":  calculated_hours,
        "avg_word_len":        avg_word_len,
        "word_count":          word_count
    })

    time.sleep(1.2)   # rest

print(f"\n Done ")
print(f"Fetched:  {len(data_list)} topics")


df = pd.DataFrame(data_list)
augmented_rows = []

for row in df.itertuples(index=False):
    for _ in range(20):
        noise = random.uniform(0.85, 1.15)
        new_hours = round(row.study_hours_needed * noise, 2)
        new_avg_len = round(row.avg_word_len * random.uniform(0.95, 1.05), 2)

        augmented_rows.append({
            "subject":            row.subject,
            "text":               row.text,
            "difficulty":         row.difficulty,
            "study_hours_needed": new_hours,
        })

large_df = pd.DataFrame(augmented_rows)

import os
os.makedirs("data", exist_ok=True)    
large_df.to_csv("data/raw_data.csv", index=False)

print(large_df[['subject', 'difficulty', 'study_hours_needed']].head())

