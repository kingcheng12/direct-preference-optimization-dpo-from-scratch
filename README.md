# Direct Preference Optimization (DPO) from Scratch

Implement Direct Preference Optimization end-to-end: log-prob utilities, a policy model, Bradley–Terry preferences, the DPO loss and gradients, IPO variants, and a full train/eval pipeline. Build the math that aligns language models to human preferences without a separate reward model or RL loop.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** log_softmax
- [x] **2.** softmax
- [x] **3.** gather_token_logprobs
- [x] **4.** masked_sequence_logprob
- [x] **5.** init_policy_params
- [x] **6.** policy_token_logits
- [x] **7.** policy_sequence_logprob
- [x] **8.** sequence_logprob_grad
- [x] **9.** bradley_terry_loss
- [x] **10.** reward_accuracy
- [x] **11.** build_preference_pairs
- [x] **12.** sample_preference_batch
- [x] **13.** freeze_reference_logprobs
- [x] **14.** policy_reference_logratio
- [x] **15.** dpo_pair_margin
- [x] **16.** dpo_loss
- [x] **17.** dpo_loss_grad
- [x] **18.** dpo_train_step
- [x] **19.** train_dpo
- [x] **20.** length_normalized_logprob
- [x] **21.** ipo_loss
- [x] **22.** implicit_reward
- [x] **23.** preference_accuracy
- [x] **24.** kl_to_reference
- [x] **25.** reward_margin_stats
- [x] **26.** evaluate_dpo
- [ ] **27.** run_dpo_pipeline

---

Built on Deep-ML.
