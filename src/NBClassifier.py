import re

import nltk

# Preprocess text
# X_train_preprocessed = np.array([text_preprocessing(text) for text in X_train])
# X_val_preprocessed = np.array([text_preprocessing(text) for text in X_val])


def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    # Uncomment to download "stopwords"
    nltk.download("stopwords")
    from nltk.corpus import stopwords

    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r"(@.*?)[\s]", " ", s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r"([\'\"\.\(\)\!\?\\\/\,])", r" \1 ", s)
    s = re.sub(r"[^\w\s\?]", " ", s)
    # Remove some special characters
    s = re.sub(r"([\;\:\|•«\n])", " ", s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join(
        [
            word
            for word in s.split()
            if word not in stopwords.words("english") or word in ["not", "can"]
        ]
    )
    # Remove trailing whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s
