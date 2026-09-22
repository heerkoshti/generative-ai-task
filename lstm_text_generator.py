"""
LSTM Text Generator
====================
Trains a word-level LSTM language model on a text corpus (default: Shakespeare)
and generates new text from a seed prompt.

Pipeline:
  1. Dataset loading & preprocessing (lowercase, strip punctuation, tokenize, n-gram sequences)
  2. Model design (Embedding -> LSTM x N -> Dense/softmax)
  3. Training (train/val split, early stopping, checkpointing)
  4. Text generation (iterative next-token prediction with temperature sampling)

Usage:
    python lstm_text_generator.py --train                     # train a model
    python lstm_text_generator.py --generate --seed "to be"   # generate from a saved model
    python lstm_text_generator.py --train --generate          # do both in one run

Dataset:
    Default corpus is the "Tiny Shakespeare" dataset (Andrej Karpathy's char-rnn repo),
    a public-domain excerpt of Shakespeare's plays:
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

    You can point --data_path to any other large .txt file (e.g. a full work downloaded
    from Project Gutenberg: https://www.gutenberg.org/).
"""

import argparse
import os
import re
import pickle
import urllib.request

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

# ---------------------------------------------------------------------------
# 1. Dataset loading & preprocessing
# ---------------------------------------------------------------------------

def download_dataset(data_path: str, url: str = DEFAULT_DATA_URL) -> str:
    """Download the default Shakespeare corpus if a local file isn't provided."""
    if os.path.exists(data_path):
        print(f"[data] Using existing file: {data_path}")
        return data_path

    print(f"[data] Downloading dataset from {url} ...")
    urllib.request.urlretrieve(url, data_path)
    print(f"[data] Saved to {data_path}")
    return data_path


def clean_text(text: str) -> str:
    """Lowercase and strip punctuation, keeping only words and whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # remove punctuation/digits
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def build_sequences(text: str, seq_length: int, tokenizer: Tokenizer):
    """
    Tokenize the corpus into word-index integers, then slide a window of
    `seq_length` tokens across the text to build (input_sequence -> next_word)
    training pairs.
    """
    token_list = tokenizer.texts_to_sequences([text])[0]

    input_sequences = []
    for i in range(seq_length, len(token_list)):
        seq = token_list[i - seq_length:i + 1]  # seq_length inputs + 1 target
        input_sequences.append(seq)

    input_sequences = np.array(input_sequences)
    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    return X, y


def load_and_preprocess(data_path: str, seq_length: int, vocab_size: int, url: str = DEFAULT_DATA_URL):
    download_dataset(data_path, url)
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned = clean_text(raw_text)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts([cleaned])

    X, y = build_sequences(cleaned, seq_length, tokenizer)

    actual_vocab_size = min(vocab_size, len(tokenizer.word_index) + 1)
    print(f"[data] Corpus length (words): {len(cleaned.split())}")
    print(f"[data] Vocabulary size: {actual_vocab_size}")
    print(f"[data] Training sequences: {X.shape[0]}")

    return X, y, tokenizer, actual_vocab_size


# ---------------------------------------------------------------------------
# 2. Model design
# ---------------------------------------------------------------------------

def build_model(vocab_size: int, seq_length: int, embedding_dim: int = 100,
                 lstm_units: int = 150, num_lstm_layers: int = 2, dropout: float = 0.2):
    """
    Embedding layer -> stacked LSTM layers -> Dense softmax output over the vocabulary.
    Using sparse_categorical_crossentropy (integer labels) instead of one-hot
    categorical_crossentropy avoids materializing a huge one-hot matrix for
    large vocabularies; it is mathematically equivalent.
    """
    model = Sequential(name="lstm_text_generator")
    model.add(Input(shape=(seq_length,)))
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))

    for i in range(num_lstm_layers):
        return_sequences = i < num_lstm_layers - 1  # only last LSTM returns a single vector
        model.add(LSTM(lstm_units, return_sequences=return_sequences))
        model.add(Dropout(dropout))

    model.add(Dense(vocab_size, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def train_model(model, X, y, epochs: int, batch_size: int, checkpoint_path: str,
                 val_split: float = 0.1):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    return history


# ---------------------------------------------------------------------------
# 4. Text generation
# ---------------------------------------------------------------------------

def sample_with_temperature(preds: np.ndarray, temperature: float = 1.0) -> int:
    """Sample the next token index from a probability distribution, with a
    temperature parameter to control randomness (lower = more conservative/
    repetitive, higher = more diverse/random)."""
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-9) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return int(np.argmax(probas))


def generate_text(model, tokenizer: Tokenizer, seed_text: str, seq_length: int,
                   num_words: int = 50, temperature: float = 0.8) -> str:
    """Iteratively predict the next word and append it to the seed text."""
    index_to_word = {idx: w for w, idx in tokenizer.word_index.items()}
    oov_index = tokenizer.word_index.get(tokenizer.oov_token)

    result = clean_text(seed_text).split()

    attempts_left = num_words * 5  # safety cap in case of repeated OOV draws
    generated = 0
    while generated < num_words and attempts_left > 0:
        attempts_left -= 1
        token_list = tokenizer.texts_to_sequences([" ".join(result)])[0]
        token_list = pad_sequences([token_list], maxlen=seq_length, padding="pre")

        preds = model.predict(token_list, verbose=0)[0]
        if oov_index is not None and oov_index < len(preds):
            preds = preds.copy()
            preds[oov_index] = 0  # never sample the <OOV> placeholder
            preds = preds / preds.sum()
        next_index = sample_with_temperature(preds, temperature)
        next_word = index_to_word.get(next_index, "")

        if not next_word:
            continue
        result.append(next_word)
        generated += 1

    return " ".join(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LSTM word-level text generator")
    parser.add_argument("--data_path", default="shakespeare.txt", help="Path to local .txt corpus (downloaded if missing)")
    parser.add_argument("--data_url", default=DEFAULT_DATA_URL, help="URL to download the corpus from if not present locally")
    parser.add_argument("--seq_length", type=int, default=20, help="Number of previous tokens used to predict the next one")
    parser.add_argument("--vocab_size", type=int, default=8000, help="Max vocabulary size")
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--lstm_units", type=int, default=150)
    parser.add_argument("--num_lstm_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model_path", default="lstm_text_model.keras")
    parser.add_argument("--tokenizer_path", default="tokenizer.pkl")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--generate", action="store_true", help="Generate text from a saved model")
    parser.add_argument("--seed", default="to be or not to", help="Seed text for generation")
    parser.add_argument("--num_words", type=int, default=50, help="Number of words to generate")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.train:
        X, y, tokenizer, vocab_size = load_and_preprocess(
            args.data_path, args.seq_length, args.vocab_size, args.data_url
        )
        model = build_model(
            vocab_size, args.seq_length, args.embedding_dim,
            args.lstm_units, args.num_lstm_layers
        )
        train_model(model, X, y, args.epochs, args.batch_size, args.model_path)
        model.save(args.model_path)
        with open(args.tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"[train] Model saved to {args.model_path}")
        print(f"[train] Tokenizer saved to {args.tokenizer_path}")

    if args.generate:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(
                f"No model found at {args.model_path}. Run with --train first."
            )
        model = load_model(args.model_path)
        with open(args.tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        generated = generate_text(
            model, tokenizer, args.seed, args.seq_length,
            args.num_words, args.temperature
        )
        print("\n[generate] Seed:", args.seed)
        print("[generate] Output:\n", generated)


if __name__ == "__main__":
    main()
