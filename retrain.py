import torch, pickle, numpy as np
from unsloth import FastLanguageModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("Loading Qwen3...")
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/qwen3-1.7b-unsloth-bnb-4bit",
    load_in_4bit=True,
    max_seq_length=128,
)
FastLanguageModel.for_inference(model)

def embed_batch(texts, batch_size=32):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
        with torch.no_grad():
            h = model(**inputs, output_hidden_states=True).hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1)
            e = ((h*mask).sum(1)/mask.sum(1)).float().cpu().numpy()
        all_embs.append(e)
    return np.vstack(all_embs)

# generate data
import random
templates = {
    "ALLOW": ["Allow user login from trusted IP", "Permit access to internal dashboard", "Allow whitelisted domain", "Grant read-only access", "Approve MFA request"],
    "BLOCK": ["Block malicious IP immediately", "Deny outbound connection to C2 server", "Block phishing domain", "Terminate suspicious process", "Drop firewall rule"],
    "QUARANTINE": ["Isolate infected endpoint", "Quarantine suspicious file for analysis", "Move device to restricted VLAN", "Contain ransomware host"],
    "FORK": ["Create ticket for manual review", "Escalate to tier-2 analyst", "Log event for investigation", "Forward to SOC lead"]
}

data, labels = [], []
for label, temps in templates.items():
    for _ in range(800):
        data.append(f"{random.choice(temps)} id{random.randint(1,99999)}")
        labels.append(label)

print(f"Embedding {len(data)} samples (batched)...")
X = embed_batch(data, batch_size=32)
y = np.array([['ALLOW','BLOCK','FORK','QUARANTINE'].index(l) for l in labels])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000, C=10, n_jobs=-1)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print(f"\nAccuracy: {acc:.3f}")

with open("export/model_fixed.pkl","wb") as f:
    pickle.dump({"clf": clf, "labels": ['ALLOW','BLOCK','FORK','QUARANTINE'], "acc": float(acc)}, f)
print("Saved to export/model_fixed.pkl")