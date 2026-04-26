import pickle, torch, warnings
from unsloth import FastLanguageModel

# silence the transformers deprecation spam
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Load the fixed model
with open("export/model_fixed.pkl", "rb") as f:
    data = pickle.load(f)

clf = data['clf']
labels = data['labels']
print(f"Loaded model (acc={data.get('acc',0):.2%}) - labels: {labels}")

# 2. Load Qwen3 once
print("Loading Qwen3-1.7B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/qwen3-1.7b-unsloth-bnb-4bit",
    load_in_4bit=True,
    max_seq_length=128,
)
FastLanguageModel.for_inference(model)

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to("cuda")
    with torch.no_grad():
        h = model(**inputs, output_hidden_states=True).hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = ((h * mask).sum(1) / mask.sum(1)).float().cpu().numpy()

    probs = clf.predict_proba(emb)[0]
    idx = probs.argmax()
    return {
        "text": text,
        "prediction": labels[idx],
        "confidence": float(probs[idx]),
        "all_scores": {labels[i]: round(float(p), 3) for i, p in enumerate(probs)}
    }

# 3. Test
tests = [
    "Block malicious IP immediately",
    "Allow the user to login from trusted network",
    "Isolate the infected endpoint and quarantine file",
    "Create a ticket for manual review by tier-2"
]

for t in tests:
    result = predict(t)
    print(f"\n{t}\n → {result['prediction']} ({result['confidence']:.0%})")
    print(f" scores: {result['all_scores']}")