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
- [ ] **6.** policy_token_logits
- [ ] **7.** policy_sequence_logprob
- [ ] **8.** sequence_logprob_grad
- [ ] **9.** bradley_terry_loss
- [ ] **10.** reward_accuracy
- [ ] **11.** build_preference_pairs
- [ ] **12.** sample_preference_batch
- [ ] **13.** freeze_reference_logprobs
- [ ] **14.** policy_reference_logratio
- [ ] **15.** dpo_pair_margin
- [ ] **16.** dpo_loss
- [ ] **17.** dpo_loss_grad
- [ ] **18.** dpo_train_step
- [ ] **19.** train_dpo
- [ ] **20.** length_normalized_logprob
- [ ] **21.** ipo_loss
- [ ] **22.** implicit_reward
- [ ] **23.** preference_accuracy
- [ ] **24.** kl_to_reference
- [ ] **25.** reward_margin_stats
- [ ] **26.** evaluate_dpo
- [ ] **27.** run_dpo_pipeline

---

Built on Deep-ML.
