import os
import pickle
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SENTIMENT_DATA_PATH = os.path.join(DATA_DIR, "sentimentdataset.csv")
MOVIE_DATA_PATH = os.path.join(DATA_DIR, "movie.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Label map for movie dataset (0 -> Negative, 1 -> Positive)
LABEL_MAP = {
    '0': 'Negative', 0: 'Negative', 'negative': 'Negative',
    '1': 'Positive', 1: 'Positive', 'positive': 'Positive',
    'neutral': 'Neutral'
}

# Step 1: Load and clean datasets
def load_sentiment_dataset(filepath):
    texts, labels = [], []
    if not os.path.exists(filepath):
        print(f"Warning: File not found at {filepath}")
        return texts, labels

    with open(filepath, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        header = {col.strip().lower(): col for col in (reader.fieldnames or [])}
        text_col = header.get("text")
        hashtag_col = header.get("hashtags")
        sentiment_col = header.get("sentiment")

        for row in reader:
            raw_sentiment = (row.get(sentiment_col) or "").strip().capitalize() if sentiment_col else ""
            if raw_sentiment in ("Positive", "Neutral", "Negative"):
                text = (row.get(text_col) or "").strip() if text_col else ""
                hashtags = (row.get(hashtag_col) or "").strip() if hashtag_col else ""
                combined_text = f"{text} {hashtags}".strip()
                if combined_text:
                    texts.append(combined_text)
                    labels.append(raw_sentiment)
    return texts, labels


def load_movie_dataset(filepath):
    texts, labels = [], []
    if not os.path.exists(filepath):
        print(f"Warning: File not found at {filepath}")
        return texts, labels

    with open(filepath, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        header = {col.strip().lower(): col for col in (reader.fieldnames or [])}
        text_col = header.get("text")
        label_col = header.get("sentiment") or header.get("label")

        for row in reader:
            text = (row.get(text_col) or "").strip() if text_col else ""
            raw_label = (row.get(label_col) or "").strip() if label_col else ""
            mapped_label = LABEL_MAP.get(raw_label) or LABEL_MAP.get(raw_label.lower()) or (raw_label.capitalize() if raw_label.capitalize() in ("Positive", "Negative", "Neutral") else None)
            
            if text and mapped_label:
                texts.append(text)
                labels.append(mapped_label)
    return texts, labels


def load_all_datasets():
    print("Loading datasets...")
    texts1, labels1 = load_sentiment_dataset(SENTIMENT_DATA_PATH)
    print(f"  - Loaded {len(texts1)} samples from sentimentdataset.csv")

    texts2, labels2 = load_movie_dataset(MOVIE_DATA_PATH)
    print(f"  - Loaded {len(texts2)} samples from movie.csv")

    all_texts = texts1 + texts2
    all_labels = labels1 + labels2

    return all_texts, all_labels


# Step 2: Extract features and labels
X, y = load_all_datasets()
print(f"\nTotal Dataset Loaded: {len(X)} samples.")
for sentiment in ("Positive", "Neutral", "Negative"):
    print(f"  - {sentiment}: {y.count(sentiment)}")

# Step 3: Evaluate model performance on 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
eval_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
X_train_vec = eval_vectorizer.fit_transform(X_train)
X_test_vec = eval_vectorizer.transform(X_test)

eval_model = LogisticRegression(class_weight="balanced", max_iter=1000)
eval_model.fit(X_train_vec, y_train)

print(f"\nHoldout Accuracy: {eval_model.score(X_test_vec, y_test):.2%}")
print(classification_report(y_test, eval_model.predict(X_test_vec), zero_division=0))

# Step 4: Train final model on full dataset
vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_vec, y)

# Step 5: Save model and vectorizer to disk
with open(os.path.join(MODELS_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("Model training complete! Artifacts saved to models/")


