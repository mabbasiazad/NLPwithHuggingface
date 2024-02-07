import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model =  AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
    ]

inputs =  tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')

inputs["labels"] = torch.tensor([1, 1])

optim = AdamW(model.parameters())
loss = model(**inputs).loss
loss.backward()
optim.step

print("your model has been trained for one batch")