from transformers import pipeline


ner_pipe = pipeline("ner", model="Gherman/bert-base-NER-Russian")
