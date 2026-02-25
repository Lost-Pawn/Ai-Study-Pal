import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/generated_mcqs.csv")

X = df['question']
Y = df['difficulty']

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_vec, Y, test_size=0.2, random_state=42
)

min_class_count = Y_train.value_counts().min()
cv_folds = min(5, min_class_count)

model = LogisticRegression()

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"],
    "max_iter": [100, 200, 300]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=cv_folds,
    scoring="accuracy"
)

grid_search.fit(X_train, Y_train)
best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
f1_accuracy = f1_score(Y_test, Y_pred, average='weighted')

print("Best Parameters:", grid_search.best_params_)
print("Accuracy:", accuracy)
print("F1 Score:", f1_accuracy)

text_vec_all = vectorizer.transform(df['question'])
predicted_labels_all = best_model.predict(text_vec_all)

predicted_difficulty_list = []
for label in predicted_labels_all:
    if label == 0:
        predicted_difficulty_list.append("easy")
    else:
        predicted_difficulty_list.append("medium")

df['predicted_difficulty'] = predicted_difficulty_list
df.to_csv("data/generated_mcqs.csv", index=False)

def generate_quiz(subject, difficulty, n_questions=10):
    subject = subject.strip().lower()
    filtered_df = df[(df['category'] == subject) & (df['difficulty'] == difficulty)]

    if filtered_df.empty:
        return None

    return filtered_df.head(n_questions)

def display_quiz(subject, difficulty, n_questions=10):
    quiz_df = generate_quiz(subject, difficulty, n_questions)

    if quiz_df is None or quiz_df.empty:
        return

    print(f"Quiz: {subject.title()} | Difficulty: {difficulty.title()}\n")

    for i, row in enumerate(quiz_df.itertuples(), 1):
        print(f"Q{i}. {row.question}")
        print(f"    A) {row.option_A}")
        print(f"    B) {row.option_B}")
        print(f"    C) {row.option_C}")
        print(f"    D) {row.option_D}")
        print(f"    Answer: {row.correct_answer}")
        print(f"    Predicted Difficulty: {row.predicted_difficulty}\n")