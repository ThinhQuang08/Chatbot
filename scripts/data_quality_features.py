import re
import pandas as pd


EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001FFFF\U00002700-\U000027BF\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000026FF\U0000FE00-\U0000FE0F]"
)

VIETNAMESE_DIACRITIC = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
    r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)


def _count_diacritics(text):
    return len(VIETNAMESE_DIACRITIC.findall(text))


def _has_emoji(text):
    return bool(EMOJI_PATTERN.search(text))


def compute_features(df):
    result = df.copy()
    text_col = "text" if "text" in result.columns else result.columns[0]
    texts = result[text_col].astype(str)

    result["text_length"] = texts.str.len()
    result["word_count"] = texts.str.split().str.len()

    letters = texts.str.replace(r"[^a-zA-ZÀ-ỹ]", "", regex=True)
    diacritics = texts.apply(_count_diacritics)
    result["diacritic_ratio"] = diacritics / letters.str.len().clip(lower=1)

    result["has_emoji"] = texts.apply(_has_emoji)
    result["is_empty"] = texts.str.strip() == ""

    char_sets = texts.str.lower().apply(set)
    unique_counts = char_sets.str.len()
    result["char_diversity"] = unique_counts / texts.str.len().clip(lower=1)

    result["ends_with_question"] = texts.str.strip().str.endswith("?")

    bool_cols = ["has_emoji", "is_empty", "ends_with_question"]
    for c in bool_cols:
        result[c] = result[c].astype(float)

    return result
