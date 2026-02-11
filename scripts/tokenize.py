import json
import argparse
import fugashi

MECABRC = "/etc/mecabrc"
MECAB_DICDIR = "/var/lib/mecab/dic/ipadic-utf8"
_KANJI_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_KANJI_UNITS = {
    "十": 10,
    "百": 100,
    "千": 1000,
}
_KANJI_LARGE_UNITS = {
    "万": 10**4,
    "億": 10**8,
}

def make_tokenizer():
    mecabrc = MECABRC
    mecab_dicdir = MECAB_DICDIR
    tagger_args = " ".join(["-r", mecabrc, "-d", mecab_dicdir])
    try:
        tagger = fugashi.Tagger(tagger_args) if tagger_args else fugashi.Tagger()
    except RuntimeError as exc:
        if "Unknown dictionary format" in str(exc):
            tagger = (
                fugashi.GenericTagger(tagger_args)
                if tagger_args
                else fugashi.GenericTagger()
            )
        else:
            raise
    return tagger

def tokenize_text(tokenizer, text):
    tokens = [token_base_form(word) for word in tokenizer(text)]
    return tokens

def token_base_form(token):
    feature = token.feature
    for attr in ("lemma", "dictionary_form", "orthBase"):
        value = getattr(feature, attr, None)
        if value and value != "*":
            return value
    return token.surface

def strip_english_gloss(token):
    if "-" not in token:
        return token
    head, tail = token.split("-", 1)
    if not head or not tail:
        return token
    return head


def kata_to_hira(text):
    if not text:
        return text
    diff = ord("ぁ") - ord("ァ")
    result = []
    for ch in text:
        code = ord(ch)
        if ord("ァ") <= code <= ord("ヶ"):
            result.append(chr(code + diff))
        else:
            result.append(ch)
    return "".join(result)


def token_kana_base(token):
    feature = token.feature
    kana = getattr(feature, "kanaBase", None) or getattr(feature, "pronBase", None)
    if not kana or kana == "*":
        kana = getattr(feature, "kana", None) or getattr(feature, "pron", None)
    if not kana or kana == "*":
        return None
    return kata_to_hira(kana)

def extract_missing_tokens_mecab(text, token_set, tagger):
    missing = []
    for token in tagger(text):
        base = strip_english_gloss(token_base_form(token))
        if base in token_set:
            continue
        print(f"Missing token base: {base}, {token.surface}")
        kana = token_kana_base(token)
        if kana and kana in token_set:
            continue
        number = kanji_to_int(base)
        if number is not None and str(number) in token_set:
            continue
        if base:
            missing.append(base)
    return missing

def kanji_to_int(text):
    if not text or any(ch not in _KANJI_DIGITS and ch not in _KANJI_UNITS and ch not in _KANJI_LARGE_UNITS for ch in text):
        return None
    total = 0
    section = 0
    number = 0
    for ch in text:
        if ch in _KANJI_DIGITS:
            number = _KANJI_DIGITS[ch]
            continue
        if ch in _KANJI_UNITS:
            unit = _KANJI_UNITS[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
            continue
        if ch in _KANJI_LARGE_UNITS:
            unit = _KANJI_LARGE_UNITS[ch]
            section += number
            if section == 0:
                section = 1
            total += section * unit
            section = 0
            number = 0
    return total + section + number

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize text data from a JSON file.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("output_file", type=str, help="Path to the output txt file. unique tokens")
    parser.add_argument("output_file_without_standard", type=str, help="Path to the output txt file. unique tokens not in standard tokens")
    parser.add_argument("--standard_tokens_file", type=str, default=None, help="Path to a file containing standard tokens (one per line).")
    args = parser.parse_args()

    if args.standard_tokens_file:
        with open(args.standard_tokens_file, "r", encoding="utf-8") as f:
            standard_tokens = set(line.strip() for line in f)

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = make_tokenizer()    
    captions = [item["caption"] for item in data["annotations"]]
    tokenized_captions = [tokenize_text(tokenizer, caption) for caption in captions]
    unique_tokens = set(token for tokens in tokenized_captions for token in tokens)
    num_unique_tokens = len(unique_tokens)
    num_captions = len(captions)
    missing_tokens = {
        token
        for caption in captions
        for token in extract_missing_tokens_mecab(caption, standard_tokens, tokenizer)
    }
    print(f"Number of captions: {num_captions}")
    print(f"Number of unique tokens: {num_unique_tokens}")
    print(f"missing tokens: {len(missing_tokens)}")
    
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        for token in sorted(unique_tokens):
            f.write(token + "\n")

    if args.standard_tokens_file:
        with open(args.output_file_without_standard, "w", encoding="utf-8") as f:
            for token in sorted(unique_tokens - standard_tokens):
                f.write(token + "\n")