"""
Sinh dữ liệu Gen Z slang (V3) — realistic Vietnamese Gen Z slang
- human_test.yml → test.yml (+ CSV) trong output_dir
- human_train.yml → train.yml (+ CSV)
- human_val.yml → val.yml (+ CSV)
- reference_normal_v3.csv → ref.csv (1000 rows)

Usage:
  python scripts/gen_genz_data.py --output-dir data/scenario3/genz_raw
"""

import re
import random
import yaml
import os
import sys
import argparse
import pandas as pd

random.seed(42)

# ============================================================
# Realistic Gen Z slang (Vietnamese teen-code)
# 
# These are actual slang used by Gen Z in Vietnam:
#   "xịn" = đẹp/tốt,    "hong" = không,   "xiền" = tiền,
#   "bao củ" = bao nhiêu,  "sống ảo" = chụp ảnh,
#   "chill" = thư giãn,   "quẩy" = vui chơi,
#   "săn" = tìm,         "dat" = ngon,     "gu" = sở thích,
#   "flex" = khoe,       "xực" = ăn,      "đỉnh" = tuyệt
# ============================================================
SLANG_MAP = {
    "đẹp": ["xịn", "xịn xò"],
    "tốt": ["xịn"],
    "xuất sắc": ["đỉnh chóp"],
    "tuyệt vời": ["đỉnh", "đỉnh chóp"],
    "tuyệt": ["đỉnh"],
    "không": ["hong"],
    "tiền": ["xiền"],
    "bao nhiêu": ["bao củ", "bao nhiu"],
    "nhiêu": ["nhiu"],
    "chụp ảnh": ["sống ảo"],
    "thư giãn": ["chill"],
    "vui chơi": ["quẩy"],
    "tìm": ["săn"],
    "ngon": ["dat"],
    "sở thích": ["gu"],
    "khoe": ["flex"],
    "ăn": ["xực"],
    "quá": ["wá"],
    # --- Extended coverage for human_test words ---
    "thế": ["thía"],
    "người": ["ngừi"],
    "luôn": ["lun"],
    "gì": ["j"],
    "rồi": ["r`"],
    "vậy": ["záy"],
    "nhé": ["nha"],
    "à": ["á"],
    "chứ": ["chớ"],
    "quen": ["wen"],
    "biết": ["bít"],
    # --- Core vocab for travel queries ---
    "đi": ["dô"],
    "nào": ["mô"],
    "muốn": ["mún"],
    "bạn": ["bồ"],
    "cũng": ["cun"],
    "được": ["đc"],
}

SLANG_ITEMS = sorted(SLANG_MAP.items(), key=lambda x: -len(x[0]))


ENTITY_PATTERN = re.compile(r'(\[[^\]]*\]\{[^}]*\})')

WORD_TRANSFORM_PROB = 0.80
SENTENCE_TRANSFORM_PROB = 0.85


def strip_entity_annotations(text):
    return ENTITY_PATTERN.sub(lambda m: m.group(1).split(']')[0][1:], text)


def preserve_entity_transform(text, transform_fn):
    parts = ENTITY_PATTERN.split(text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            result.append(transform_fn(part))
        else:
            result.append(part)
    return ''.join(result)


def apply_genz_v3(text, force_change=False):
    if not isinstance(text, str) or not text.strip():
        return text
    if not force_change and random.random() > SENTENCE_TRANSFORM_PROB:
        return text

    changed = False
    for standard, slang_list in SLANG_ITEMS:
        if random.random() < WORD_TRANSFORM_PROB:
            slang = random.choice(slang_list)
            new_text = re.sub(
                r'\b' + re.escape(standard) + r'\b',
                slang,
                text,
                flags=re.IGNORECASE,
                count=1
            )
            if new_text != text:
                text = new_text
                changed = True

    # Force at least one change if requested
    if force_change and not changed:
        available = [(s, sl) for s, sl in SLANG_ITEMS
                      if re.search(r'\b' + re.escape(s) + r'\b', text, flags=re.IGNORECASE)]
        if available:
            standard, slang_list = random.choice(available)
            slang = random.choice(slang_list)
            text = re.sub(
                r'\b' + re.escape(standard) + r'\b',
                slang,
                text,
                flags=re.IGNORECASE,
                count=1
            )
            changed = True

    text = re.sub(r'\s+', ' ', text)
    return text


def parse_nlu_yml(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    data = yaml.safe_load(content)
    rows = []
    for item in data.get('nlu', []):
        intent = item.get('intent', '')
        examples = item.get('examples', '')
        for line in examples.strip().split('\n'):
            line = line.strip()
            if line.startswith('- '):
                text = line[2:].strip()
                rows.append((intent, text))
    return rows


def write_nlu_yml(rows, output_path):
    from collections import OrderedDict
    groups = OrderedDict()
    for intent, text in rows:
        groups.setdefault(intent, []).append(text)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('version: "3.1"\n\nnlu:\n')
        for intent, examples in groups.items():
            f.write(f'  - intent: {intent}\n')
            f.write(f'    examples: |\n')
            for ex in examples:
                f.write(f'      - {ex}\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='data/scenario3/genz_raw',
                        help='Output directory for generated files')
    parser.add_argument('--test-variants', type=int, default=5,
                        help='Number of Gen Z variants per test example (default: 5)')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    out_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- Process test set ---
    test_rows = parse_nlu_yml(os.path.join(data_dir, 'human_test.yml'))
    print(f"[GENZ-V3] human_test.yml: {len(test_rows)} examples (variants={args.test_variants})")

    # Generate multi-variant test set from human_test (all unique, no reference supplement)
    test_genz = []
    for i, t in test_rows:
        for v in range(args.test_variants):
            t_genz = preserve_entity_transform(t, lambda txt: apply_genz_v3(txt, force_change=True))
            test_genz.append((i, t_genz))

    # Deduplicate
    seen = set()
    test_genz_dedup = []
    for i, t in test_genz:
        key = (i, strip_entity_annotations(t))
        if key not in seen:
            seen.add(key)
            test_genz_dedup.append((i, t))
    test_genz = test_genz_dedup

    print(f"  (final: {len(test_genz)} unique examples)")

    write_nlu_yml(test_genz, os.path.join(out_dir, 'test.yml'))
    df_test = pd.DataFrame(
        [(strip_entity_annotations(t), i) for i, t in test_genz],
        columns=['text', 'intent']
    )
    df_test.to_csv(os.path.join(out_dir, 'test.csv'),
                   index=False, encoding='utf-8-sig')

    print(f"  → {len(test_genz)} examples → CSV + YAML")

    # --- Process human_train.yml ---
    train_rows = parse_nlu_yml(os.path.join(data_dir, 'human_train.yml'))
    print(f"[GENZ-V3] human_train.yml: {len(train_rows)} examples")

    train_genz = [(i, preserve_entity_transform(t, apply_genz_v3))
                   for i, t in train_rows]
    write_nlu_yml(train_genz, os.path.join(out_dir, 'train.yml'))
    df_train = pd.DataFrame(
        [(strip_entity_annotations(t), i) for i, t in train_genz],
        columns=['text', 'intent']
    )
    df_train.to_csv(os.path.join(out_dir, 'train.csv'),
                    index=False, encoding='utf-8-sig')

    n_changed_train = sum(1 for i, t in train_genz
                          if t != strip_entity_annotations(dict((i2, t2) for i2, t2 in train_rows).get(i, '')))
    print(f"  → {len(train_genz)} examples, ~{n_changed_train} changed → CSV + YAML")

    # --- Process human_val.yml ---
    val_rows = parse_nlu_yml(os.path.join(data_dir, 'human_val.yml'))
    print(f"[GENZ-V3] human_val.yml: {len(val_rows)} examples")

    val_genz = [(i, preserve_entity_transform(t, apply_genz_v3))
                 for i, t in val_rows]
    write_nlu_yml(val_genz, os.path.join(out_dir, 'val.yml'))
    df_val = pd.DataFrame(
        [(strip_entity_annotations(t), i) for i, t in val_genz],
        columns=['text', 'intent']
    )
    df_val.to_csv(os.path.join(out_dir, 'val.csv'),
                  index=False, encoding='utf-8-sig')

    n_changed_val = sum(1 for i, t in val_genz
                        if t != strip_entity_annotations(dict((i2, t2) for i2, t2 in val_rows).get(i, '')))
    print(f"  → {len(val_genz)} examples, ~{n_changed_val} changed → CSV + YAML")

    # --- Process reference_normal_v3.csv (for TextDrift amplification) ---
    ref_path = os.path.join(data_dir, 'reference_normal_v3.csv')
    ref_genz = []
    if os.path.exists(ref_path):
        df_ref = pd.read_csv(ref_path)
        df_sample = df_ref.sample(n=1000, random_state=42)
        for _, row in df_sample.iterrows():
            t = str(row['text'])
            t_genz = apply_genz_v3(t)
            ref_genz.append((t_genz, row['intent']))
        df_ref_out = pd.DataFrame(ref_genz, columns=['text', 'intent'])
        csv_path = os.path.join(out_dir, 'ref.csv')
        df_ref_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[GENZ-V3] reference 1000 → CSV: {csv_path}")

    # Print samples
    print("\n[GENZ-V3] === MẪU SO SÁNH (10 mẫu có thay đổi) ===")
    count = 0
    for i in range(len(test_rows)):
        orig = test_rows[i][1]
        genz_text = test_genz[i][1]
        if orig != genz_text and count < 10:
            print(f"  GỐC:  {orig}")
            print(f"  GENZ: {genz_text}")
            print()
            count += 1

    # Count actual transformations
    all_genz = test_genz + train_genz + val_genz + ref_genz
    stats = {k: 0 for k in SLANG_MAP}
    for _, text in all_genz:
        for standard in SLANG_MAP:
            for slang in SLANG_MAP[standard]:
                if slang in text.lower():
                    stats[standard] += 1
    print("\n[GENZ-V3] Thống kê slang đã dùng:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        if v > 0:
            print(f"  {k:15s} → {str(SLANG_MAP[k]):20s} x{v} lần")


if __name__ == '__main__':
    main()
