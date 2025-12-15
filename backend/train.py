import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys
import os

# Add current directory to path to import preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_text

def train():
    print("Loading dataset...")
    # Adjust path if necessary. Assuming script is run from backend/ or root
    csv_path = '../training.1600000.processed.noemoticon.csv'
    if not os.path.exists(csv_path):
        csv_path = 'training.1600000.processed.noemoticon.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    column_names = ['target', 'id', 'date', 'flag' , 'user' , 'text']
    # Read a subset for faster demonstration, remove nrows for full training
    # twitter_data = pd.read_csv(csv_path, names=column_names, encoding='ISO-8859-1', nrows=50000)
    twitter_data = pd.read_csv(csv_path, names=column_names, encoding='ISO-8859-1')
    
    print("Dataset loaded. Shape:", twitter_data.shape)

    # Replace target 4 with 1
    twitter_data.replace({'target': {4:1}}, inplace=True)

    print("Preprocessing text (this may take a while)...")
    twitter_data['processed_content'] = twitter_data['text'].apply(preprocess_text)

    X = twitter_data['processed_content'].values
    Y = twitter_data['target'].values

    print("Splitting data...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=300000, sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    print("Training model...")
    model = LogisticRegression(max_iter=1000, C=2.0, solver="liblinear", class_weight="balanced")
    model.fit(X_train, Y_train)

    print("Evaluating model...")
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
    print(f"Training Accuracy: {training_data_accuracy}")

    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
    print(f"Test Accuracy: {test_data_accuracy}")

    print("Saving artifacts...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    with open(os.path.join(output_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Done! Model and vectorizer saved to {output_dir}")

if __name__ == "__main__":
    train()
