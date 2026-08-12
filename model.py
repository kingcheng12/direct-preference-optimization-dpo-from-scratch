"""
Direct Preference Optimization (DPO) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - log_softmax
def log_softmax(logits, axis=-1):
    # TODO: convert logits into numerically stable log-probabilities along axis
    
    shifted_logits = logits - np.max(logits, axis = axis, keepdims=True)

    log_probs = shifted_logits - np.log(np.sum(np.exp(shifted_logits), axis = axis, keepdims=True))

    return log_probs

# Step 2 - softmax
def softmax(logits, axis=-1):
    # TODO: Convert an array of logits into a probability distribution along a given axis
    
    shifted_logits = logits - np.max(logits, axis = axis, keepdims = True)
    exp_logits = np.exp(shifted_logits)

    return exp_logits / np.sum(exp_logits, axis = axis, keepdims = True)

# Step 3 - gather_token_logprobs
def gather_token_logprobs(log_probs, token_ids):
    # TODO: Extract the log-probability of each observed token from a full vocab log-prob tensor...

    expanded_token_ids = token_ids[..., None]

    gathered_log_probs = np.take_along_axis(
        log_probs,
        expanded_token_ids,
        axis=-1,
    )

    token_log_probs = gathered_log_probs[..., 0]

    return token_log_probs

# Step 4 - masked_sequence_logprob
def masked_sequence_logprob(token_logprobs, mask):
    # TODO: Sum per-token log-probabilities under a binary mask to obtain a single sequence log-probability per example.
    
    masked_token_logprobs = np.where(mask, token_logprobs, 0)

    return np.sum(masked_token_logprobs, axis = -1)

# Step 5 - init_policy_params
def init_policy_params(vocab_size, d_model, rng=None):
    # TODO: Initialize the policy language-model parameters with small random values

    if rng is None:
        rng = np.random.default_rng()
    
    embed = rng.normal(loc=0, scale=0.02, size = (vocab_size, d_model))
    W_out = rng.normal(loc=0, scale=0.02, size = (d_model, vocab_size))
    b_out = np.zeros(vocab_size)

    return {'embed': embed, 'W_out': W_out, 'b_out': b_out}

# Step 6 - policy_token_logits
def policy_token_logits(params, token_ids):
    # TODO: Compute next-token logits for every position from policy params and token ids.
    embed = params['embed']
    W_out = params['W_out']
    b_out = params['b_out']

    # select embedding based on token ids
    # (B, T) to (B, T, hidden_dim)
    hidden = embed[token_ids]

    return hidden @ W_out + b_out

# Step 7 - policy_sequence_logprob
def policy_sequence_logprob(params, token_ids, mask):
    # TODO: Compute the total masked sequence log-probability under the current policy...

    logits = policy_token_logits(params, token_ids)

    log_probs = log_softmax(logits, axis=-1)

    token_log_probs = gather_token_logprobs(log_probs, token_ids)

    sequence_log_probs = masked_sequence_logprob(token_log_probs, mask)

    return sequence_log_probs

# Step 8 - sequence_logprob_grad
def sequence_logprob_grad(params, token_ids, mask):
    # TODO: Compute gradients of the summed sequence log-probability w.r.t. params
    embed = params['embed']
    W_out = params['W_out']
    b_out = params['b_out']

    # forward
    hidden = embed[token_ids]     
    logits = hidden @ W_out + b_out

    log_probs = log_softmax(logits, axis=-1)
    probs = np.exp(log_probs) 

    grad_logits = -probs
    # backward

    # d log p(target) / d logits = one_hot(target) - probabilities
    batch_indices = np.arange(token_ids.shape[0])[:, None]
    time_indices = np.arange(token_ids.shape[1])[None, :]

    grad_logits[
        batch_indices,
        time_indices,
        token_ids,
    ] += 1.0

    grad_logits *= mask[..., None]
    grad_W_out = np.einsum(
        'btd,btv->dv',
        hidden,
        grad_logits,
    )
    grad_b_out = np.sum(
        grad_logits,
        axis=(0, 1),
    )
    grad_hidden = grad_logits @ W_out.T
    grad_embed = np.zeros_like(embed)
    np.add.at(
        grad_embed,
        token_ids,
        grad_hidden,
    )

    return {
        'embed': grad_embed,
        'W_out': grad_W_out,
        'b_out': grad_b_out,
    }

# Step 9 - bradley_terry_loss
def bradley_terry_loss(reward_chosen, reward_rejected):
    # TODO: Compute the mean Bradley-Terry pairwise preference loss...
    
    logits = reward_chosen - reward_rejected

    scores = np.log(1 + np.exp(-logits))

    return np.mean(scores, axis = -1)

# Step 10 - reward_accuracy
def reward_accuracy(reward_chosen, reward_rejected):
    # TODO: Fraction of pairs where chosen reward is strictly higher than rejected.
    
    return np.mean(reward_chosen > reward_rejected)

# Step 11 - build_preference_pairs
def build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    # TODO: Package raw arrays into a list of preference-pair dictionaries

    pairs = []

    for i in range(len(prompts)):
        pairs.append({
            'prompt': prompts[i],
            'chosen_ids': chosen_ids[i],
            'rejected_ids': rejected_ids[i],
            'chosen_mask': chosen_mask[i],
            'rejected_mask': rejected_mask[i],
        })

    return pairs

# Step 12 - sample_preference_batch
def sample_preference_batch(pairs, batch_size, rng=None):
    # TODO: Sample a mini-batch of preference pairs for one training step.
    
    if rng is None:
        # generate rng
        rng = np.random.default_rng()

    n = len(pairs)
    replace = True if batch_size > n else False

    indices = rng.choice(n, size = batch_size, replace = replace)

    batch = {
        'chosen_ids': np.stack([pairs[i]['chosen_ids'] for i in indices]),
        'rejected_ids': np.stack([pairs[i]['rejected_ids'] for i in indices]),
        'chosen_mask': np.stack([pairs[i]['chosen_mask'] for i in indices]),
        'rejected_mask': np.stack([pairs[i]['rejected_mask'] for i in indices]),
    }

    # Include prompts if they are present.
    if 'prompt' in pairs[0]:
        batch['prompt'] = np.array(
            [pairs[i]['prompt'] for i in indices]
        )

    return batch

# Step 13 - freeze_reference_logprobs
def freeze_reference_logprobs(ref_params, pairs):
    # TODO: Precompute and freeze reference-model sequence log-probabilities for every chosen and rejected response...

    out = []

    for pair in pairs:
        chosen_logprob = policy_sequence_logprob(
            ref_params,
            pair['chosen_ids'][None, :],
            pair['chosen_mask'][None, :],
        )[0]

        rejected_logprob = policy_sequence_logprob(
            ref_params,
            pair['rejected_ids'][None, :],
            pair['rejected_mask'][None, :],
        )[0]

        new_pair = pair.copy()
        new_pair['chosen'] = chosen_logprob
        new_pair['rejected'] = rejected_logprob

        out.append(new_pair)

    return out

# Step 14 - policy_reference_logratio
def policy_reference_logratio(policy_logprob, reference_logprob):
    # TODO: Compute the per-sequence log-ratio log pi_theta(y) - log pi_ref(y)
    return policy_logprob - reference_logprob

# Step 15 - dpo_pair_margin
def dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: Compute the scaled DPO pair margin for a batch of preference pairs

    # Policy-vs-reference 
    chosen_logratio = policy_logprob_chosen - ref_logprob_chosen

    # Policy-vs-reference log-ratio for rejected response
    rejected_logratio = policy_logprob_rejected - ref_logprob_rejected

    # DPO preference margin
    margin = beta * (chosen_logratio - rejected_logratio)

    return margin

# Step 16 - dpo_loss
def dpo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: return the mean logistic loss on the DPO pair margins as a scalar float
    
    margins = dpo_pair_margin(
        policy_logprob_chosen,
        policy_logprob_rejected,
        ref_logprob_chosen,
        ref_logprob_rejected,
        beta,
    )

    # -log(sigmoid(margin))
    losses = np.logaddexp(0.0, -margins)

    return float(np.mean(losses))

# Step 17 - dpo_loss_grad
def dpo_loss_grad(params, batch, ref_logprobs_batch, beta):
    # TODO: Evaluate DPO loss and return parameter gradients for the policy

    policy_chosen = policy_sequence_logprob(
        params,
        batch['chosen_ids'],
        batch['chosen_mask'],
    )

    policy_rejected = policy_sequence_logprob(
        params,
        batch['rejected_ids'],
        batch['rejected_mask'],
    )

    # Frozen reference log-probabilities
    ref_chosen = ref_logprobs_batch['chosen']
    ref_rejected = ref_logprobs_batch['rejected']

    # DPO margins
    margins = dpo_pair_margin(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta,
    )

    # Mean DPO loss
    loss = float(
        np.mean(np.logaddexp(0.0, -margins))
    )

    B = len(margins)

    # d[-log sigmoid(m)] / dm
    # = -sigmoid(-m)
    grad_margin = -1.0 / (1.0 + np.exp(margins))
    grad_margin /= B

    chosen_weights = beta * grad_margin
    rejected_weights = -beta * grad_margin

    # Weight each sequence's contribution to the parameter gradients.
    chosen_grads = sequence_logprob_grad(
        params,
        batch['chosen_ids'],
        batch['chosen_mask'] * chosen_weights[:, None],
    )

    rejected_grads = sequence_logprob_grad(
        params,
        batch['rejected_ids'],
        batch['rejected_mask'] * rejected_weights[:, None],
    )

    grads = {
        key: chosen_grads[key] + rejected_grads[key]
        for key in params
    }

    return loss, grads

# Step 18 - dpo_train_step
import numpy as np

def dpo_train_step(params, batch, ref_logprobs_batch, beta, learning_rate):
    # TODO: Execute one DPO gradient-descent update; return updated params + metrics

    # Compute DPO loss and gradients
    loss, grads = dpo_loss_grad(
        params,
        batch,
        ref_logprobs_batch,
        beta,
    )

    # Gradient-descent update
    new_params = {
        name: params[name] - learning_rate * grads[name]
        for name in params
    }

    # training metrics
    metrics = {
        'loss': float(loss),
    }

    return new_params, metrics

# Step 19 - train_dpo
def train_dpo(params, pairs, ref_logprobs, beta, learning_rate, num_steps, batch_size, rng=None):
    # TODO: Sample batches, run DPO train steps, record per-step metrics.

    # Train the policy with DPO for num_steps

    if rng is None:
        rng = np.random.default_rng()

    history = []
    n = len(pairs)

    for step in range(num_steps):
        indices = rng.choice(
            n,
            size=batch_size,
            replace=(batch_size > n),
        )

        batch = {
            'chosen_ids': np.stack([
                np.asarray(pairs[i]['chosen_ids'])
                for i in indices
            ]),
            'rejected_ids': np.stack([
                np.asarray(pairs[i]['rejected_ids'])
                for i in indices
            ]),
            'chosen_mask': np.stack([
                np.asarray(pairs[i]['chosen_mask'])
                for i in indices
            ]),
            'rejected_mask': np.stack([
                np.asarray(pairs[i]['rejected_mask'])
                for i in indices
            ]),
        }

        ref_batch = {
            'chosen': np.asarray(ref_logprobs['chosen'])[indices],
            'rejected': np.asarray(ref_logprobs['rejected'])[indices],
        }

        params, metrics = dpo_train_step(
            params,
            batch,
            ref_batch,
            beta,
            learning_rate,
        )

        metrics = dict(metrics)
        metrics['step'] = step

        history.append(metrics)

    return params, history

# Step 20 - length_normalized_logprob
def length_normalized_logprob(seq_logprob, mask):
    # TODO: Normalize sequence log-probabilities by their valid token counts.
    
    lengths = np.sum(mask, axis=-1)
    lengths = np.maximum(lengths, 1.0)

    return seq_logprob / lengths

# Step 21 - ipo_loss
def ipo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    # TODO: Evaluate mean squared IPO loss on unscaled log-ratio margins

    chosen_logratio = (
        policy_logprob_chosen - ref_logprob_chosen
    )

    rejected_logratio = (
        policy_logprob_rejected - ref_logprob_rejected
    )

    margin = chosen_logratio - rejected_logratio

    target = 1.0 / (2.0 * beta)

    losses = (margin - target) ** 2

    return float(np.mean(losses))

# Step 22 - implicit_reward (not yet solved)
# TODO: implement

# Step 23 - preference_accuracy (not yet solved)
# TODO: implement

# Step 24 - kl_to_reference (not yet solved)
# TODO: implement

# Step 25 - reward_margin_stats (not yet solved)
# TODO: implement

# Step 26 - evaluate_dpo (not yet solved)
# TODO: implement

# Step 27 - run_dpo_pipeline (not yet solved)
# TODO: implement

