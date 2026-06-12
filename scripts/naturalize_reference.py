"""
Naturalize reference_normal_v3.csv — thay template starters bằng natural alternatives
để loại bỏ style gap với human data.

Usage:
  .venv/bin/python scripts/naturalize_reference.py
  -> Output: data/reference_natural_v4.csv
"""

import pandas as pd
import re
import random

random.seed(42)

# Vietnamese place names that need proper casing
PLACE_NAMES = {
    'Hà Nội', 'Sài Gòn', 'Đà Nẵng', 'Đà Lạt', 'Nha Trang',
    'Phú Quốc', 'Hạ Long', 'Hội An', 'Huế', 'Mũi Né', 'Sa Pa', 'Sapa',
    'Vũng Tàu', 'Hải Phòng', 'Cần Thơ', 'Phan Thiết',
    'Kim Liên', 'Hồ Chí Minh', 'Bà Nà', 'Ba Vì',
    'Cát Bà', 'Cửa Lò', 'Đồ Sơn', 'Phong Nha', 'Tam Cốc',
    'Tràng An', 'Bích Động', 'Chùa Hương', 'Yên Tử',
}

def fix_proper_nouns(text):
    """Restore proper casing for known place names.
    Only matches lowercase version of name to avoid double-capitalizing.
    """
    for name in PLACE_NAMES:
        name_lower = name.lower()
        text = re.sub(r'\b' + re.escape(name_lower) + r'\b', name, text)
    return text


STARTER_MAP = {
    # "cho tôi hỏi X" → various natural forms
    r'^[Cc]ho tôi hỏi\s+': [
        '',                                    # direct: "X có ... không?"
        'mình muốn ',                          # "mình muốn tìm tour..."
        'có ',                                 # "có tour nào ... không?"
        'tìm giúp ',                            # "tìm giúp mình..."
        'cho mình hỏi ',                       # "cho mình hỏi..."
        'mình cần tìm ',                       # "mình cần tìm..."
    ],
    # "cho em hỏi X" → various natural forms
    r'^[Cc]ho em hỏi\s+': [
        '',
        'em muốn ',
        'có ',
        'cho mình ',
        'mình cần ',
    ],
    # "làm ơn X" → various natural forms
    r'^[Ll]àm ơn\s+': [
        '',
        'làm ơn cho mình ',
        'giúp mình ',
        'nhờ bạn ',
        'phiền bạn ',
    ],
    # "ad ơi X" → various natural forms
    r'^[Aa]d ơi[,\s]+': [
        '',
        'bạn ơi ',
        'cho mình hỏi ',
        'admin ơi ',
        '',
    ],
    # "ad ơi" (no comma)
    r'^[Aa]d ơi\s+': [
        '',
        'bạn ơi ',
        'cho mình hỏi ',
        '',
    ],
    # "mình hỏi X" → various
    r'^[Mm]ình hỏi\s+': [
        '',
        'mình muốn hỏi ',
        'cho mình hỏi ',
        '',
    ],
}

# Additional transformations to make text more natural
ADDITIONAL_FIXES = [
    # "có ... không" patterns (add "nào" or restructure)
    (r'\bcó\s+(.+?)\s+không\s*$', lambda m: f'có {m.group(1)} không' if random.random() < 0.5 else f'{m.group(1)} có không'),
    # Fix capitalization after replacement
    (r'^[a-z]', lambda m: m.group(0).upper()),  # first letter uppercase
]


def naturalize(text):
    original = text
    # Apply starter replacements
    for pattern, alternatives in STARTER_MAP.items():
        if re.search(pattern, text):
            replacement = random.choice(alternatives)
            text = re.sub(pattern, replacement, text, count=1)
            break  # only apply one replacement

    if not text.strip():
        text = original

    # Lowercase everything
    text = text.strip().lower()

    # Restore proper noun casing (matches lowercase patterns)
    text = fix_proper_nouns(text)

    # Capitalize first letter of sentence
    if text:
        text = text[0].upper() + text[1:]

    return text


def main():
    df = pd.read_csv('data/reference_normal_v3.csv')
    print(f'Input: {len(df)} rows')

    # Naturalize each text
    df['text'] = df['text'].apply(naturalize)

    # Remove duplicates (may happen after naturalization)
    before = len(df)
    df = df.drop_duplicates(subset=['text'])
    print(f'After dedup: {len(df)} rows ({before - len(df)} duplicates removed)')

    # Verify diversity improved
    starters_before = ['cho tôi hỏi', 'cho em hỏi', 'làm ơn', 'ad ơi', 'mình hỏi']
    for s in starters_before:
        cnt = sum(1 for t in df.text if str(t).strip().lower().startswith(s))
        print(f'  Still starts with \"{s}\": {cnt} ({100*cnt/len(df):.1f}%)')

    # Check top starters now
    from collections import Counter
    starters = [str(t).strip().split()[0] if len(str(t).strip().split()) > 0 else '' for t in df.text]
    top10 = Counter(starters).most_common(10)
    print(f'\nTop 10 starters after naturalization:')
    for w, c in top10:
        print(f'  \"{w}\": {c} ({100*c/len(df):.1f}%)')

    df.to_csv('data/reference_natural_v4.csv', index=False, encoding='utf-8-sig')
    print(f'\nOutput: data/reference_natural_v4.csv ({len(df)} rows)')


if __name__ == '__main__':
    main()
