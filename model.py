"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
def clean_text(text: str) -> str:
    # TODO: Lowercase text and replace non-alphabetic chars with spaces
    text_lower = text.lower()
    text_cleaned = "".join(t if t.isalpha() else " " for t in text_lower)

    return text_cleaned.strip()

# Step 2 - tokenize
def tokenize(text: str) -> list:
    # TODO: Split cleaned text on whitespace into non-empty word tokens
    return text.split()

# Step 3 - tokenize_corpus
def tokenize_corpus(texts: list) -> list:
    # TODO: Apply clean_text and tokenize to every document so the full corpus becomes a list of token lists.
    token_list = []
    for text in texts: 
        cleaned_text = clean_text(text)
        token_list.append(tokenize(cleaned_text))
    return token_list

# Step 4 - split_train_val_test_indices
def split_train_val_test_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int = 0) -> tuple:
    # TODO: Produce shuffled index arrays that partition n_samples into train/val/test
    rng = np.random.default_rng(seed)
    
    samples = np.arange(0, n_samples)
    rng.shuffle(samples)

    n_val = int(val_fraction * n_samples)
    n_test = int(test_fraction * n_samples)
    n_train = n_samples - n_val - n_test

    train_idx = samples[:n_train]
    val_idx = samples[n_train: n_train + n_val]
    test_idx = samples[n_train + n_val:]

    return train_idx, val_idx, test_idx

# Step 5 - count_word_frequencies
def count_word_frequencies(tokenized_docs: list) -> dict:
    count_word = {} #word: count

    for doc in tokenized_docs: 
        for word in doc:
            count_word[word] = 1 + count_word.get(word, 0)

    return count_word

# Step 6 - build_vocabulary
def build_vocabulary(word_counts: dict, max_size: int) -> dict:
   V = min(max_size, len(word_counts))

   kept_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))[:V]

   return {word: i for i, (word, _) in enumerate(kept_words)}

# Step 7 - tokens_to_bow
def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    # TODO: Convert one document's token list into a bag-of-words count vector...
    token_count = {}
    out = [0.0] * len(vocab)

    for token in tokens: 
        token_count[token] = 1.0 + token_count.get(token, 0.0)
    
    for word, idx in vocab.items():
        if word in token_count: 
            out[idx] = token_count[word]
    return np.array(out)

# Step 8 - corpus_to_bow_matrix
def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    # TODO: Stack per-document BoW vectors into a 2-D count matrix for a whole corpus.
    n = len(tokenized_docs)
    v = len(vocab)

    out = np.zeros((n ,v))
    count_docs = {} #idx_doc:{word, count}
    for idx_doc in range(n):
        count_docs[idx_doc] = {}

        for word in tokenized_docs[idx_doc]:
            count_docs[idx_doc][word] = 1 + count_docs[idx_doc].get(word, 0)
    
    for idx_doc in range(n):
        for word, i in vocab.items():
            if word in count_docs[idx_doc]:
                out[idx_doc][i] = count_docs[idx_doc][word]
    return out

# Step 9 - compute_document_frequencies
def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    # TODO: Count docs where each term appears at least once (df, shape (V,))
    v = bow_matrix.shape[-1]
    out = np.zeros(v, dtype=int)

    for bow_row in bow_matrix:
        for i in range(v): 
            if bow_row[i] > 0: 
                out[i] += 1
    return out

# Step 10 - compute_idf
def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    # TODO: Compute smoothed IDF idf_j = log((n_docs + 1) / (df_j + 1)) + 1
    v = len(df)
    out = np.zeros(v)

    for i in range(v): 
        idf = np.log((n_docs + 1)/(df[i] + 1)) + 1
        out[i] = idf
    return out

# Step 11 - transform_tfidf
def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    # TODO: Multiply BoW counts by the fitted IDF vector to produce TF-IDF features.
    return bow_matrix * idf

# Step 12 - fit_tfidf
def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    df = compute_document_frequencies(bow_train)
    return compute_idf(df, bow_train.shape[0])

# Step 13 - sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    # TODO: Map logits to probabilities with a numerically stable logistic sigmoid.
    return 1 / (1+np.exp(-z))

# Step 14 - logistic_predict_proba
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    # TODO: Return P(y=1|x) for each row via linear scores and sigmoid
    probs = np.dot(X, w) + b
    return sigmoid(probs)

# Step 15 - binary_cross_entropy
def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> float:

    binary_cross_entropy_error = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    
    return np.mean(binary_cross_entropy_error) +  (l2_lambda * np.sum(w**2)) / 2

# Step 16 - logistic_gradients
def logistic_gradients(X: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> tuple:
    """Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.

    Args:
        X: Feature matrix of shape (N, D).
        y_true: Binary labels of shape (N,).
        y_prob: Predicted probabilities of shape (N,).
        w: Weight vector of shape (D,).
        l2_lambda: L2 regularization strength.

    Returns:
        Tuple (dw, db) with dw shape (D,) and db a float.
    """ 
    N = np.shape(X)[0]
    dw = X.T @ (y_prob - y_true) / N + l2_lambda * w
    db =  np.mean(y_prob - y_true)

    return (dw, db)

# Step 17 - initialize_logistic_params
def initialize_logistic_params(n_features: int) -> tuple:
    
    return (np.zeros(n_features), 0.0)

# Step 18 - gradient_descent_step
def gradient_descent_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float, l2_lambda: float) -> tuple:

    y_prob =logistic_predict_proba(X, w, b)
    loss = binary_cross_entropy(y, y_prob, w, l2_lambda)
    dw, db = logistic_gradients(X, y, y_prob, w, l2_lambda)

    w_new = -lr * dw + w
    b_new = -lr * db + b

    return (w_new, b_new, loss)

# Step 19 - train_logistic_regression
def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, l2_lambda: float, n_epochs: int) -> tuple:
    (w, b) = initialize_logistic_params(np.shape(X)[-1])
    losses = []
    for _ in range(n_epochs):
        w_new, b_new, loss =gradient_descent_step(X, y, w, b, lr, l2_lambda)
        w, b = w_new, b_new
        losses.append(loss)

    return (w, b, losses)

# Step 20 - predict_labels
def predict_labels(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into hard binary labels.

    Args:
        proba: 1-D array of probabilities in [0, 1], shape (N,).
        threshold: Decision threshold; proba >= threshold maps to 1.

    Returns:
        Integer array of shape (N,) with values in {0, 1}.
    """
    return np.array([1 if p >= threshold else 0 for p in proba])

# Step 21 - confusion_counts
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] and y_pred[i]:
            tp += 1
        elif y_true[i] and not y_pred[i]:
            fn += 1
        elif not y_true[i] and y_pred[i]:
            fp += 1
        else:
            tn += 1
    return (tp, fp, tn, fn)

# Step 22 - metrics_from_counts
def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    return {'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy)}

# Step 23 - tune_decision_threshold
def tune_decision_threshold(y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray = None) -> tuple:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    best_threshold = None
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = predict_labels(proba, threshold)

        tp, fp, tn, fn = confusion_counts(y_true, y_pred)

        metrics = metrics_from_counts(tp, fp, tn, fn)
        f1 = metrics["f1"]

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold), float(best_f1)

# Step 24 - evaluate_predictions
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    (tp, fp, tn, fn) = confusion_counts(y_true, y_pred)
    metrics = metrics_from_counts(tp, fp, tn, fn)

    return {
        'tp': tp,
        'fp': fp, 
        'tn': tn, 
        'fn': fn, 
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'accuracy': metrics['accuracy']       
    }

# Step 25 - vectorize_texts
def vectorize_texts(texts: list, vocab: dict, idf: np.ndarray) -> np.ndarray:    
    tokenized = tokenize_corpus(texts)
    bow = corpus_to_bow_matrix(tokenized, vocab)

    return transform_tfidf(bow, idf)

# Step 26 - predict_text
def predict_text(text: str, vocab: dict, idf: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> int:
    """Label a single raw message with the fitted classifier.

    Args:
        text: Raw input string.
        vocab: Fitted word -> column index map.
        idf: Fitted IDF vector, shape (V,).
        w: Logistic weight vector, shape (V,).
        b: Logistic bias scalar.
        threshold: Decision threshold for the positive class.

    Returns:
        Predicted label as int 0 or 1.
    """

    X = vectorize_texts([text], vocab, idf)
    proba = logistic_predict_proba(X, w, b) 
    label = predict_labels(proba, threshold)

    return label[0]

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

