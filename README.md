# Study Helper System - Flask Application

This is a Flask-based AI-powered learning platform that processes educational content and generates study materials.

## Features

1. **Wikipedia Data Fetcher** - Fetches educational content from Wikipedia
2. **Data Cleaning** - Cleans and processes raw data
3. **MCQ Generator** - Generates multiple choice questions using NLP
4. **ML Model Training** - Trains model to predict question difficulty
5. **Cluster Analysis** - Groups topics and provides learning resources
6. **Study Tips Generator** - Generates personalized study tips

## Installation

```bash
pip install flask pandas numpy wikipedia-api nltk scikit-learn matplotlib tensorflow
```

## Required Downloads

Some features require additional data:
- For text summarization: Download GloVe embeddings (glove.6B.50d.txt) and place in `data/` folder

## Running the Application

```bash
python app.py
```

Then open your browser to: http://localhost:5000

## File Structure

```
/
├── app.py                  # Main Flask application
├── templates/
│   ├── index.html         # Main interface
│   └── view_data.html     # Data viewing page
├── static/
│   └── style.css          # Basic styles
└── data/                  # Generated data files
    ├── raw_data.csv
    ├── clean_data.csv
    ├── generated_mcqs.csv
    └── study_tips.csv
```

## Usage

1. **Step 1**: Click "Fetch Data" to download Wikipedia content
2. **Step 2**: Click "Clean Data" to process the raw data
3. **Step 3**: Click "Generate MCQs" to create questions
4. **Step 4**: Click "Train Model" to train the ML model
5. **Step 5**: Click "Run Clustering" to group topics
6. **Step 6**: Click "Generate Tips" to get study tips
7. Use "Get Quiz" to retrieve questions by subject/difficulty
8. Use "View Data" to see generated datasets
9. Use "Download Files" to download CSV files

## Note

The text summarization feature from code #6 requires TensorFlow and GloVe embeddings which are large files. This has been simplified in the Flask app but can be added if needed.

## Technologies Used

- Flask (Web Framework)
- Pandas (Data Processing)
- Scikit-learn (Machine Learning)
- NLTK (Natural Language Processing)
- Wikipedia API (Content Fetching)
- Matplotlib (Data Visualization)
