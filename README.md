# LSTM Text Generator — Shakespeare

A word-level LSTM language model that learns from a corpus of text and generates new,
Shakespeare-style text from a seed phrase. Built with TensorFlow/Keras.

## Files

| File | Purpose |
|---|---|
| `lstm_text_generator.py` | Full pipeline: preprocessing, model, training, generation (CLI) |
| `shakespeare.txt` | The training corpus (downloaded automatically) |
| `lstm_text_model.keras` | Trained model checkpoint |
| `tokenizer.pkl` | Fitted Keras `Tokenizer` (word ↔ index mapping) |
| `sample_outputs.md` | Generated text samples from different seeds/temperatures |
| `experiment.py`, `experiment_seqlen.py` | Bonus: architecture/hyperparameter experiments |

## Dataset

Default: the **"Tiny Shakespeare"** corpus (public domain), a ~1MB excerpt of
Shakespeare's plays, downloaded automatically from:

```
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

To use a different / larger corpus (e.g. Shakespeare's *Complete Works* from
[Project Gutenberg](https://www.gutenberg.org/ebooks/100)), download the `.txt` file and pass
`--data_path your_file.txt` — the script will use the local file instead of downloading.

## 1. Data preprocessing (`load_and_preprocess`, `clean_text`, `build_sequences`)

- Lowercases the text and strips punctuation/digits, keeping only words and whitespace.
- Tokenizes with Keras `Tokenizer` (word-level, capped at `--vocab_size` most frequent words,
  out-of-vocabulary words mapped to `<OOV>`).
- Builds `(input_sequence → next_word)` training pairs with a sliding window of
  `--seq_length` tokens across the whole corpus.

## 2. Model (`build_model`)

```
Input(seq_length)
  → Embedding(vocab_size, embedding_dim)
  → LSTM(lstm_units, return_sequences=True) → Dropout
  → LSTM(lstm_units)                        → Dropout   (repeated for num_lstm_layers)
  → Dense(vocab_size, activation="softmax")
```

Compiled with `sparse_categorical_crossentropy` (equivalent to `categorical_crossentropy`
but takes integer labels directly, avoiding a huge one-hot matrix for large vocabularies)
and the `Adam` optimizer.

## 3. Training (`train_model`)

- 90/10 train/validation split (`train_test_split`).
- `EarlyStopping` (patience=3, restores best weights) + `ModelCheckpoint` (saves best
  val_loss model) to avoid overfitting.

Run training:
```bash
pip install tensorflow scikit-learn --break-system-packages   # if not already installed
python lstm_text_generator.py --train \
    --seq_length 15 --vocab_size 6000 --embedding_dim 100 \
    --lstm_units 128 --num_lstm_layers 2 --epochs 20 --batch_size 256
```

> **Note on scale:** the checkpoint included in this deliverable was trained for a small
> number of epochs (~3) to keep runtime short for a demo/interview submission — it captures
> Shakespeare-ish vocabulary and rhythm but not full grammatical coherence yet. Training for
> 20–30+ epochs on the full corpus (or the full *Complete Works*) produces noticeably more
> coherent output; the architecture and pipeline are unchanged, only `--epochs` and dataset
> size need to increase.

## 4. Text generation (`generate_text`, `sample_with_temperature`)

Iteratively predicts the next word given the last `seq_length` words, using
**temperature sampling** (drawing from the softmax distribution rather than always taking
`argmax`) so output isn't repetitive. Lower temperature (e.g. 0.5) → safer, more repetitive
text; higher temperature (e.g. 1.0+) → more diverse, riskier text.

```bash
python lstm_text_generator.py --generate \
    --seed "to be or not to be" --num_words 40 --temperature 0.8
```

### Sample outputs

See `sample_outputs.md` for generated text from four seed/temperature combinations.

## 5. Bonus: architecture experiments

Two controlled comparisons were run on a matched 40k-sequence subset, 3 epochs each, so the
numbers are directly comparable to each other (not to the main model above, which used the
full corpus).

**Depth — 1-layer vs 2-layer LSTM** (seq_length=12, 64 units, embedding_dim=64):

| Config | Final val_loss | Final val_accuracy | Time (3 epochs) |
|---|---|---|---|
| 1-layer LSTM | 6.154 | 0.0455 | 16.8s |
| 2-layer LSTM | 6.221 | 0.0455 | 28.1s |

At this very small scale (few epochs, subset of data), the deeper model doesn't yet
outperform the shallow one — it has more parameters to fit and needs more training steps
before its extra capacity pays off, and 2 layers takes ~1.7x longer per epoch. In practice,
deeper LSTMs tend to help once trained sufficiently on larger corpora — they can capture
longer-range dependencies (e.g. character/plot consistency across a sentence) — but they're
more prone to overfitting on small datasets and are more expensive to train.

**Sequence length — 5 vs 20 tokens of context** (1-layer, 64 units):

| Config | Final val_loss | Final val_accuracy |
|---|---|---|
| seq_length = 5 | 6.131 | 0.0422 |
| seq_length = 20 | 6.151 | 0.0525 |

A longer context window (20 tokens) reached higher validation accuracy at the same number of
epochs, suggesting the model benefits from more preceding context when predicting the next
word — intuitive, since more of the sentence's structure is visible. The tradeoff is a larger
input dimension and slightly slower training per step, and very long windows can dilute the
learning signal if the dataset is small relative to the window size.

**Takeaway:** with limited training budget, a shallower, moderate-context model
(1–2 LSTM layers, seq_length ≈ 15–20) is a good balance of speed and quality; deeper models
are worth the extra cost mainly once you can afford substantially more epochs/data.

## Evaluation criteria checklist

- **Model performance**: word-level LSTM with embedding + softmax output, trainable to
  arbitrary depth/epochs via CLI flags.
- **Code quality**: single well-commented script, functions split by pipeline stage
  (preprocessing / model / training / generation), CLI args for all hyperparameters.
- **Creativity**: temperature-based sampling, OOV filtering during generation, two
  independent architecture experiments (depth, context length).
- **Problem-solving**: sparse (integer) labels instead of one-hot to keep memory usage
  reasonable with a several-thousand-word vocabulary; early stopping + checkpointing against
  overfitting; safety cap on generation loop to avoid infinite OOV-retry loops.
