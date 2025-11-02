from transformers import AutoModelForSequenceClassification

def build_model(model_name="bert-base-uncased", num_labels=2):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
