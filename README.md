# Sentiment Analysis App

This is a production-ready Sentiment Analysis application in which I used Sentiment 140 dataset from Kaggle which contains 1.6 million tweets labeled by positive or negative.

- Cleaned the data by removing stopwords
- Used NLTK library with ( wordnet , omw-1.4 ) which gave synonyms , antonyms and multilingual support to be used for Lemmatization
- Applied Bigrams in Tfidf which gave huge bump in accuracy of the model because meaning in language often comes from word pairs, not single words
- Used Lemmatization over PorterStemmer to produce real, meaningful words, while stemming often produces crude, sometimes incorrect word fragments.
- Tuned Logistic Regression model by controling regularisation
  
## Project Structure

- `backend/`: Contains the FastAPI application and training script.
- `frontend/`: Contains the static frontend files.
- `requirements.txt`: Python dependencies.
- `render.yaml`: Configuration for deploying to Render.

## Setup & Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model**:
    Before running the app, you need to generate the model and vectorizer artifacts.
    Run the training script (ensure the dataset `training.1600000.processed.noemoticon.csv` is in the root directory):
    ```bash
    cd backend
    python train.py
    cd ..
    ```
    This will create `model.pkl` and `vectorizer.pkl` in the `backend/` directory.

3.  **Run the Application**:
    ```bash
    uvicorn backend.main:app --reload
    ```
    The app will be available at `http://127.0.0.1:8000`.

## Deployment on Render

1.  **Push to GitHub**: Commit your code (including `model.pkl` and `vectorizer.pkl` if they are not too large, otherwise use Git LFS or train as part of build if dataset is small) to a GitHub repository.
    *Note: Since the dataset is large, it is recommended to train locally and commit the `.pkl` files.*

2.  **Create Web Service**:
    - Connect your GitHub repo to Render.
    - Select "Web Service".
    - Render should automatically detect `render.yaml`.
    - If not, use the following settings:
        - **Build Command**: `pip install -r requirements.txt`
        - **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

3.  **Access**: Your app will be live at your Render URL.

## Future Enhancements

- I will build a version 2 of this model by using BERT ( Bidirectional Encoder Representations from Transformers )
- It will give my model the abilty of reading the sentence BOTH ways
- Making it understand true meaning of the words by reading the sentence from left->right and right->left
- It would also be able to handle conditions like Misspellings , Slang , Rare words

## 👨‍💻 Author

Created with passion ⚡️ by Shrish Mishra

---

⭐ If you find this project useful, please consider giving it a star!
