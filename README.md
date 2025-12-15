# Sentiment Analysis App

This is a production-ready Sentiment Analysis application using Logistic Regression and made it's API using FastAPI and a simple HTML/CSS/JS frontend.

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
