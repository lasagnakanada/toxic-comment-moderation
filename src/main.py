import json
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path

from src.predictor import Predictor
from src.config import settings


# ---- PREDICTOR INSTANCE ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(settings.THRESHOLDS_PATH) as f:
    best_thresholds = json.load(f)
tokenizer = DistilBertTokenizer.from_pretrained(settings.BERT_MODEL_NAME)
bert_model = DistilBertModel.from_pretrained(settings.BERT_MODEL_NAME).to(device)
predictor_instance = Predictor(
    model_path=settings.MODEL_PATH,
    thresholds=best_thresholds,
    tokenizer=tokenizer,
    bert_model=bert_model,
    input_size=settings.INPUT_SIZE,
    hidden_size=settings.HIDDEN_SIZE,
    output_size=settings.OUTPUT_SIZE,
    device=device,
)

app = FastAPI(
    title="Toxic Comment Classifier API",
    description="An API to classify text for toxicity.",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    probs: list[float]
    labels: dict[str, bool]

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    """
    Accepts a text string and returns a JSON object with toxicity probabilities and labels.
    """
    probs, labels = predictor_instance.predict(request.text)
    return {"probs": probs, "labels": labels}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Serves the home page with a form to enter a comment.
    """
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/classify", response_class=HTMLResponse)
async def classify(request: Request, text: str = Form(...)):
    """
    Accepts a form submission, classifies the text, and returns an HTML results page.
    """
    probs, labels_dict = predictor_instance.predict(text)
    
    # Combine results for templating
    results = [
        (label, prob, labels_dict[label])
        for label, prob in zip(settings.LABELS, probs)
    ]
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "text": text,
        "results": results
    })