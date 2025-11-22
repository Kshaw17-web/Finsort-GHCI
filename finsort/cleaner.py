import re

SPECIAL_WORDS = {
    "mktplace": "marketplace",
    "dept": "department",
    "co": "company",
    "pvt": "private",
    "ltd": "limited"
}

def clean_transaction(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove common bank/upi codes like 'sq *', 'upi', 'vps', etc.
    text = re.sub(r'sq \*', ' ', text)
    text = re.sub(r'\b(upi|vps|rtgs|imps|pmts|trx|swipe|pay)\b', ' ', text)
    # remove non-letters (keep spaces)
    text = re.sub(r'[^a-z ]', ' ', text)
    # collapse spaces
    text = ' '.join(text.split())
    # normalize - use word boundaries to avoid partial matches
    words = text.split()
    normalized_words = []
    for word in words:
        if word in SPECIAL_WORDS:
            normalized_words.append(SPECIAL_WORDS[word])
        else:
            normalized_words.append(word)
    text = ' '.join(normalized_words)
    return text
