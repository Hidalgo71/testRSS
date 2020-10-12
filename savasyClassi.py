import os
import pandas as pd

print(os.listdir())
df = pd.read_csv("E:/7allV03.csv", )
print(df.head())

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("savasy/bert-turkish-text-classification")
model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-turkish-text-classification")

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
nlp("bla bla")
