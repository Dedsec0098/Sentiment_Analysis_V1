from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
import os
import sys
from fastapi.middleware.cors import CORSMiddleware


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_text

app = FastAPI(title="Sentiment Analysis API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
vectorizer = None

def load_model_and_vectorizer():
    global model, vectorizer
    try:
  
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print("Warning: Model or vectorizer not found. Please run train.py first.")
            return False

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
            
        print("Model and vectorizer loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return False

@app.on_event("startup")
def startup_event():
    load_model_and_vectorizer()

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    global model, vectorizer
    if model is None or vectorizer is None:
        # Attempt to reload
        if not load_model_and_vectorizer():
            raise HTTPException(status_code=503, detail="Model not loaded. Please run training script.")


    processed_text = preprocess_text(request.text)
    

    vectorized_text = vectorizer.transform([processed_text])
    

    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text).max()
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return {"sentiment": sentiment, "confidence": float(probability)}


frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
