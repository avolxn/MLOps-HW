import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

model = AutoModelForSequenceClassification.from_pretrained(
    "avolxn/disaster-classification-bert"
)
tokenizer = AutoTokenizer.from_pretrained("avolxn/disaster-classification-bert")

model.eval()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    inputs = tokenizer(
        text, return_tensors="pt", max_length=100, truncation=True, padding="max_length"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).item()

    return jsonify({"disaster": predictions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
