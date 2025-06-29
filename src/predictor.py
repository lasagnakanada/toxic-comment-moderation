import json
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from src.model_definition import ToxicClassifier
from src.config import settings

class Predictor:
    def __init__(
        self,
        model_path,
        thresholds,
        tokenizer,
        bert_model,
        input_size,
        hidden_size,
        output_size,
        device
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.classifier = ToxicClassifier(input_size, hidden_size, output_size)
        self.classifier.load_state_dict(torch.load(model_path, map_location=device))
        self.classifier.to(device)
        self.classifier.eval()
        self.bert_model.eval()
        self.best_thresholds = thresholds 

    def predict(self, text: str) -> tuple[list[float], dict[str, bool]]:
        """Performs inference on a single text string."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=settings.MAX_LENGTH
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
            output = self.classifier(embedding)
            probs = torch.sigmoid(output).cpu().numpy().tolist()[0]

        preds = [prob > thr for prob, thr in zip(probs, self.best_thresholds)]
        label_predictions = {label: bool(pred) for label, pred in zip(settings.LABELS, preds)}
        return probs, label_predictions