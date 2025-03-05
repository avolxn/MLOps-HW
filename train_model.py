import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

f1_metric = evaluate.load("f1")
recall_metric = evaluate.load("recall")
precision_metric = evaluate.load("precision")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_PATH = "./nlp-getting-started"


def load_and_preprocess_data():
    train_dataset = pd.read_csv(f"{INPUT_PATH}/train.csv").drop(
        columns=["keyword", "location", "id"]
    )

    train_df, val_df = train_test_split(train_dataset, test_size=0.2, random_state=42)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    return train_ds, val_ds


def tokenize(batch):
    return tokenizer(
        batch["text"], max_length=100, truncation=True, padding="max_length"
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    precision = precision_metric.compute(predictions=predictions, references=labels)[
        "precision"
    ]

    return {"eval_f1": f1, "eval_recall": recall, "eval_precision": precision}


def load_model_and_tokenizer():
    model_name = "microsoft/deberta-v3-large"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


train_ds, val_ds = load_and_preprocess_data()
model, tokenizer = load_model_and_tokenizer()

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("target", "labels")
val_ds = val_ds.rename_column("target", "labels")
train_ds = train_ds.remove_columns(["text"])
val_ds = val_ds.remove_columns(["text"])

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=30,
    save_strategy="epoch",
    num_train_epochs=3,
    max_steps=150,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("./saved_model")
tokenizer.save_pretrained("./saved_model")
