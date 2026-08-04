"""
Direct Preference Optimization (DPO) from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo of Direct Preference Optimization (DPO) with synthetic data.

Runs the preference-math helpers on hand-picked log-probs, then the full
pipeline on a LEARNABLE preference dataset (chosen responses use a "good" token
range, rejected use a disjoint "bad" range) so training visibly improves the
policy -- preference accuracy climbs toward 1.0 and the reward margin widens.
"""
import numpy as np


def main():
    np.random.seed(0)
    beta = 0.1

    # --- Core preference math on synthetic log-probs (no model required) ---
    pol_c = np.array([-2.0, -3.1, -1.4, -2.5])
    pol_r = np.array([-3.5, -2.0, -2.8, -1.9])
    ref_c = np.array([-2.2, -3.0, -1.6, -2.4])
    ref_r = np.array([-3.2, -2.3, -2.6, -2.1])
    print("DPO pair margins:", np.round(dpo_pair_margin(pol_c, pol_r, ref_c, ref_r, beta), 4))
    print("DPO loss:", float(np.round(dpo_loss(pol_c, pol_r, ref_c, ref_r, beta), 6)))
    print("IPO loss:", float(np.round(ipo_loss(pol_c, pol_r, ref_c, ref_r, beta), 6)))
    print("preference accuracy (toy):", float(preference_accuracy(pol_c, pol_r, ref_c, ref_r, beta)))
    print("reward margin stats (toy):", reward_margin_stats(pol_c, pol_r, ref_c, ref_r, beta))

    # --- Full pipeline on a LEARNABLE dataset ---
    rng = np.random.default_rng(0)
    vocab_size = 12
    d_model = 8
    n_pairs = 16
    seq_len = 6
    half = vocab_size // 2
    prompts = rng.integers(0, vocab_size, size=(n_pairs, 3))
    chosen_ids = rng.integers(0, half, size=(n_pairs, seq_len))            # "good" tokens
    rejected_ids = rng.integers(half, vocab_size, size=(n_pairs, seq_len))  # "bad" tokens
    chosen_mask = np.ones((n_pairs, seq_len))
    rejected_mask = np.ones((n_pairs, seq_len))

    result = run_dpo_pipeline(
        vocab_size, d_model, prompts, chosen_ids, rejected_ids,
        chosen_mask, rejected_mask,
        beta=0.1, learning_rate=0.3, num_steps=300, batch_size=8,
        rng=rng,
    )
    hist = result["history"]
    ev = result["eval_metrics"]
    print("")
    print("DPO training: loss", round(hist[0]["loss"], 4), "->", round(hist[-1]["loss"], 4))
    print("eval preference_accuracy:", round(float(ev["preference_accuracy"]), 4))
    print("eval mean reward margin :", round(float(ev["mean_margin"]), 4))
    print("eval dpo_loss           :", round(float(ev["dpo_loss"]), 4))


if __name__ == "__main__":
    main()

