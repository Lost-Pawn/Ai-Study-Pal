import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DSA_TOPICS = {
    "introduction to data structures", "arrays and strings", "linked lists",
    "stacks and queues", "trees and binary trees", "binary search trees",
    "heaps and priority queues", "graphs and graph traversals",
    "hashing and hash tables", "recursion and backtracking",
    "sorting algorithms", "searching algorithms", "dynamic programming",
    "greedy algorithms", "divide and conquer techniques",
    "trie and prefix trees", "union-find (disjoint set union)",
    "segment trees and fenwick trees", "kd-trees and ball trees", "skip lists"
}

PROGRAMMING_TOPICS = {
    "introduction to programming", "variables and data types",
    "operators and expressions", "control flow statements",
    "loops: for, while, nested loops", "functions and recursion",
    "object-oriented programming concepts", "classes and objects",
    "inheritance and polymorphism", "encapsulation and abstraction",
    "exception handling", "file i/o operations", "modules and packages",
    "debugging and testing", "version control with git",
    "concurrency and multithreading", "asynchronous programming with asyncio",
    "type annotations and static checking"
}

CLOUD_TOPICS = {
    "introduction to cloud engineering",
    "cloud service models: iaas, paas, saas",
    "public, private, and hybrid clouds", "virtual machines and containers",
    "storage services in the cloud", "networking in cloud platforms",
    "cloud security and identity management", "monitoring and logging",
    "serverless computing", "scaling applications and load balancing",
    "deploying applications to aws/azure/gcp",
    "cloud architecture best practices",
    "cost optimization and resource management",
    "disaster recovery in cloud",
    "ci/cd pipelines and automation in cloud",
    "kubernetes for ml orchestration", "serverless ml inference"
}

def map_subject(topic):
    topic = topic.strip().lower()
    if topic in DSA_TOPICS:
        return "dsa"
    if topic in PROGRAMMING_TOPICS:
        return "programming"
    if topic in CLOUD_TOPICS:
        return "cloud engineering"
    return topic

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

df = pd.read_csv("data/generated_mcqs.csv")
df["category"] = df["subject"].apply(map_subject)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

texts_cluster = df[["category", "paragraph"]].copy()
X_cluster = vectorizer.fit_transform(texts_cluster["paragraph"])
cluster_samples = X_cluster.shape[0]

if cluster_samples >= 2:
    n_clusters = min(5, cluster_samples)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    texts_cluster["cluster"] = kmeans.fit_predict(X_cluster)
else:
    texts_cluster["cluster"] = 0

df["cluster"] = texts_cluster["cluster"]

cluster_map = texts_cluster.groupby("cluster")["category"].agg(lambda x: x.value_counts().idxmax()).to_dict()

def get_resources_by_cluster(cluster_id):
    category = cluster_map.get(cluster_id, "")
    return category, resources_links.get(category, [])

def get_resources_by_subject(subject_name):
    category = map_subject(subject_name.strip().lower())
    cluster_id = None
    for cid, cat in cluster_map.items():
        if cat == category:
            cluster_id = cid
            break
    return cluster_id, resources_links.get(category, [])

for cluster_id in sorted(cluster_map):
    category, resources = get_resources_by_cluster(cluster_id)
    print(f"Cluster {cluster_id} | Category: {category}")
    for r in resources:
        print(f"   {r}")

df.to_csv("data/generated_mcqs.csv", index=False)