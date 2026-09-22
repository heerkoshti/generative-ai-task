import time
from lstm_text_generator import load_and_preprocess, build_model, train_model

for seq_len in [5, 20]:
    X, y, tokenizer, vocab_size = load_and_preprocess("shakespeare.txt", seq_length=seq_len, vocab_size=4000)
    X, y = X[:40000], y[:40000]
    print("="*60, "seq_length =", seq_len)
    model = build_model(vocab_size, seq_length=seq_len, embedding_dim=64, lstm_units=64, num_lstm_layers=1)
    hist = train_model(model, X, y, epochs=3, batch_size=256, checkpoint_path=f"exp_seq{seq_len}.keras")
    print(f"RESULT seq_length={seq_len}: val_loss={hist.history['val_loss'][-1]:.4f} val_acc={hist.history['val_accuracy'][-1]:.4f}")
