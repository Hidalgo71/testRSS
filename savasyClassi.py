import os
import pandas as pd

print(os.listdir())
df = pd.read_csv("E:/eval.csv", )
print(df.head())

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, \
    AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("savasy/bert-turkish-text-classification")
model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-turkish-text-classification")

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print(nlp("bla bla"))

code_to_label = {
    'LABEL_0': 'dunya ',
    'LABEL_1': 'ekonomi ',
    'LABEL_2': 'kultur ',
    'LABEL_3': 'saglik ',
    'LABEL_4': 'siyaset ',
    'LABEL_5': 'spor ',
    'LABEL_6': 'teknoloji '}

var = code_to_label[nlp("bla bla")[0]['label']]

print(df.head())

pr=nlp(df.text[20])
print(pr)

var = pr[0]['label']
var = code_to_label[pr[0]['label']]

preds=nlp(list([t[:300] for t in df.text]))
preds_codes=[int(p['label'].split("_")[1]) for p in preds]


print(preds_codes)
print("#")
print(list(df.labels))

sum(preds_codes ==df.labels) / len(preds)