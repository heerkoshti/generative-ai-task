# Sample Generated Text

Model: 2-layer LSTM (128 units, embedding_dim=100), seq_length=15, vocab_size=6000,
trained ~3 epochs on the Tiny Shakespeare corpus.

---

**Seed:** `"to be or not to be"` (temperature=0.8)

> to be or not to be a state then a father good lord in our troth to rome by god in fire the
> lord for my very father must or now the sound play come comes

---

**Seed:** `"the king said"` (temperature=0.5, more conservative)

> the king said then these name s art of the king of do the s lord of two my good son a liest
> a tale is the own

---

**Seed:** `"the king said"` (temperature=1.0, more diverse)

> the king said wither shall with a fairest fault to cannot cony beseech or dear true called
> us them thy smiling faith thee paulina it the adventure and

---

**Seed:** `"love is"` (temperature=0.5)

> love is he the father and a queen and be will a lord in the day and the way of the life and
> he the life and

---

**Seed:** `"love is"` (temperature=1.0)

> love is shall the mind he the own soul coriolanus hands you have forbid benvolio are the
> cat queen now quiet so saw is this rage a

---

### Observations

- Even after only ~3 epochs, the model has clearly picked up Shakespearean *vocabulary*
  (thee, wither, beseech, coriolanus, benvolio) and short archaic-sounding phrase fragments
  ("a fairest fault", "smiling faith").
- Grammatical coherence is still weak — sentences don't reliably parse — which is expected
  at this training scale. Training longer (20–30 epochs) on the full corpus is expected to
  substantially improve fluency, since validation loss was still decreasing when training
  stopped.
- Lower temperature (0.5) output is visibly more repetitive ("the ... and the ... and")
  while higher temperature (1.0) is more lexically varied but a bit less controlled — the
  classic diversity/coherence tradeoff of temperature sampling.
