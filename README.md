# Codenames Hint Finder

A small project that chooses the best single-word hint for the board game Codenames. Given a board (from a screenshot) and the current team's target words, the notebook computes embeddings, finds the centroid (average direction) of your team's words, pushes that direction away from other words, and picks the nearest word from a vocabulary as the hint.

Files
- [Spymaster.ipynb](Spymaster.ipynb) — main pipeline and functions (OCR, embedding, scoring, candidate selection).
- [Operative.ipynb](Operative.ipynb) — helper/operative-side utilities.
- [google-10000-english-usa.txt](google-10000-english-usa.txt) — source vocabulary used for hint candidates. ([https://gist.github.com/h3xx/1976236](https://github.com/first20hours/google-10000-english?tab=readme-ov-file))
- [google_word_embeddings.pkl](google_word_embeddings.pkl) — precomputed embeddings for the vocab (can be re-created with [`save_vocab_embeddings`](Spymaster.ipynb)).
- [requirements.txt](requirements.txt) — Python dependencies.
- sc_*.png — example board screenshots (e.g. [sc_9.png](sc_9.png)).

How it works (brief)
1. OCR: read board words from screenshot using EasyOCR (in [Spymaster.ipynb](Spymaster.ipynb)).
2. Embed: words → normalized embeddings using a SentenceTransformer via [`embed`](Spymaster.ipynb).
3. Compute target direction: centroid of your team's word embeddings.
4. Penalize links to opponent/neutral/assassin words and compute a score for each candidate in the vocab.
5. Rank candidates and simulate how each candidate ranks the board words; pick the hint that yields the longest initial correct-run using [`score_choices_order`](Spymaster.ipynb) and [`score_candidates`](Spymaster.ipynb). Final selection via [`pick_best_candidate`](Spymaster.ipynb).

Scoring formula
The score used for candidate ranking is (conceptually):

```math
\text{score} = {mean}(\text{sim\_{good}}) - \lambda_1  \text{sim\_bad\_max} - \lambda_2  \text{sim\_bad\_mean}
```

where similarities are dot-products between normalized embeddings and $\lambda_1$, $\lambda_2$ are tunable weights.

Quick start
1. Install deps:
```sh
pip install -r requirements.txt
```
