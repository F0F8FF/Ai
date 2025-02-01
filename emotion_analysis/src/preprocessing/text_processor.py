from transformers import AutoTokenizer

class TextProcessor:
    def __init__(self, model_name="bert-base-multilingual-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def process(self, text):
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ) 