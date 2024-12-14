from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

model_id = '/home/ubuntu/LLM/deberta-v3-base-prompt-injection-v2'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)


def llm_guard_v2(query):
    output = classifier(query)[0]['label']
    
    if output == 'SAFE':
        return 0
    return 1


if __name__ == '__main__':
    print(llm_guard_v2('hahaha'))