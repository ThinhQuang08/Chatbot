import yaml, random, os

random.seed(42)

NLU_PATH = "/home/thinh/Chatbot_tien/rasa_bot/data/train/nlu.yml"

with open(NLU_PATH, encoding="utf-8") as f:
    data = yaml.safe_load(f)

all_examples = {}
for item in data["nlu"]:
    if "intent" not in item:
        continue
    intent = item["intent"]
    examples = item.get("examples", "")
    texts = []
    for line in examples.strip().split("\n"):
        line = line.strip()
        if line.startswith("- "):
            texts.append(line[2:])
    all_examples.setdefault(intent, []).extend(texts)

print(f"Intents: {len(all_examples)}")
print(f"Examples: {sum(len(v) for v in all_examples.values())}")

# Split 80:10:10
train, val, test = {}, {}, {}
for intent, texts in all_examples.items():
    random.shuffle(texts)
    n = len(texts)
    n_train = max(1, int(n * 0.8))
    n_val = max(0, int(n * 0.1))
    n_train_actual = min(n_train, n)
    n_val_actual = min(n_val, n - n_train_actual)
    train[intent] = texts[:n_train_actual]
    val[intent] = texts[n_train_actual:n_train_actual + n_val_actual]
    test[intent] = texts[n_train_actual + n_val_actual:]

print(f"Train: {sum(len(v) for v in train.values())} {len(train)} intents")
print(f"Val:   {sum(len(v) for v in val.values())} {len(val)} intents")
print(f"Test:  {sum(len(v) for v in test.values())} {len(test)} intents")

def write_nlu(groups, path):
    lines = ["# Auto-split from nlu.yml", 'version: "3.1"', "", "nlu:"]
    for intent in sorted(groups.keys()):
        if not groups[intent]:
            continue
        lines.append(f"  - intent: {intent}")
        lines.append("    examples: |")
        for t in groups[intent]:
            lines.append(f"      - {t}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  -> {path} ({sum(len(v) for v in groups.values())} ex)")

base = "/home/thinh/Chatbot_tien/data"
write_nlu(train, os.path.join(base, "human_train.yml"))
write_nlu(val, os.path.join(base, "human_val.yml"))
write_nlu(test, os.path.join(base, "human_test.yml"))

trainval = {}
for intent in set(list(train.keys()) + list(val.keys())):
    trainval[intent] = train.get(intent, []) + val.get(intent, [])
write_nlu(trainval, os.path.join(base, "human_trainval.yml"))
