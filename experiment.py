import time, sys
sys.path.insert(0, ".")
from lstm_text_generator import load_and_preprocess, build_model, train_model

X, y, tokenizer, vocab_size = load_and_preprocess("shakespeare.txt", seq_length=12, vocab_size=4000)
# use a subset for a fast, fair comparison
N = 40000
X, y = X[:N], y[:N]

configs = [
    {"name": "1-layer LSTM (64 units)", "num_lstm_layers": 1, "lstm_units": 64},
    {"name": "2-layer LSTM (64 units)", "num_lstm_layers": 2, "lstm_units": 64},
]

results = []
for cfg in configs:
    print("="*60, cfg["name"])
    t0 = time.time()
    model = build_model(vocab_size, seq_length=12, embedding_dim=64,
                         lstm_units=cfg["lstm_units"], num_lstm_layers=cfg["num_lstm_layers"])
    hist = train_model(model, X, y, epochs=3, batch_size=256, checkpoint_path=f"exp_{cfg['num_lstm_layers']}layer.keras")
    dt = time.time() - t0
    final_val_loss = hist.history["val_loss"][-1]
    final_val_acc = hist.history["val_accuracy"][-1]
    results.append((cfg["name"], final_val_loss, final_val_acc, dt))

print("\n\nSUMMARY")
for name, vl, va, dt in results:
    print(f"{name}: val_loss={vl:.4f} val_acc={va:.4f} time={dt:.1f}s")
