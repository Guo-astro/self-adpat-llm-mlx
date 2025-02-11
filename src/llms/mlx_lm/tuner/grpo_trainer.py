# Copyright © 2024 Apple Inc.
from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .trainer import grad_checkpoint, TrainingArgs, TrainingCallback, average_gradients, iterate_batches


@dataclass
class GRPOTrainingArgs(TrainingArgs):
    group_size: int = field(
        default=4,
        metadata={"help": "Number of responses per prompt."},
    )
    beta: float = field(
        default=0.1, metadata={"help": "KL penalty coefficient."}
    )
    epsilon: float = field(
        default=1e-4, metadata={"help": "The Epsilon for numerical stability."}
    )
    max_completion_length: int = field(
        default=512, metadata={"help": "Number of Generations."}
    )
    reference_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to reference model weights. If None, uses the same model."
        }
    )


def r1_extract_xml_answer(text: str) -> str:
    """Extracts the answer from an XML formatted text string."""
    try:
        answer: str = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except Exception:
        print("r1_extract_xml_answer returned empty string")
        return ""


def r1_int_reward_func(prompts: List[str], completions: List[Optional[str]], answer: List[str], **kwargs: Any) -> List[
    float]:
    """Ensures we always return a list of floats."""
    if not completions:
        return [0.0] * len(prompts)
    extracted_responses: List[str] = [r1_extract_xml_answer(r) for r in completions if r is not None]
    return [0.5 if r and r.isdigit() else 0.0 for r in extracted_responses]


def r1_accuracy_reward_func(prompts: List[str], completions: List[Optional[str]], answer: List[str], **kwargs: Any) -> \
List[float]:
    """Ensures we always return a list of floats."""
    if not completions or not answer:
        return [0.0] * len(prompts)
    extracted_responses: List[str] = [r1_extract_xml_answer(r) for r in completions if r is not None]
    return [2.0 if r and a and r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def r1_soft_format_reward_func(prompts: List[str], completions: List[Optional[str]], answer: List[str],
                               **kwargs: Any) -> List[float]:
    """Ensures we always return a list of floats."""
    if not completions:
        return [0.0] * len(prompts)
    pattern: str = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches: List[bool] = [bool(re.search(pattern, r)) if r else False for r in completions]
    return [0.5 if match else 0.0 for match in matches]


def r1_strict_format_reward_func(prompts: List[str], completions: List[Optional[str]], answer: List[str],
                                 **kwargs: Any) -> List[float]:
    """Ensures we always return a list of floats."""
    if not completions:
        return [0.0] * len(prompts)
    pattern: str = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches: List[bool] = [bool(re.search(pattern, r)) if r else False for r in completions]
    return [0.5 if match else 0.0 for match in matches]


def r1_count_xml(prompts: List[str], completions: List[Optional[str]], answer: List[str], **kwargs: Any) -> List[float]:
    """Ensures we always return a list of floats."""
    if not completions:
        return [0.0] * len(prompts)

    scores: List[float] = []
    for text in completions:
        if not text:
            scores.append(0.0)
            continue

        count: float = 0.0
        if text.count("<think>\n") == 1:
            count += 0.125
        if text.count("\n</think>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
        if text.count("\n</answer>\n") == 1:
            count += 0.125

        # Penalize extra text after </answer>
        end_text: str = text.split("\n</answer>\n")[-1]
        count -= len(end_text) * 0.001 if len(end_text) > 0 else 0

        scores.append(max(0.0, count))  # Ensure non-negative score

    return scores


def generate_grpo(model: Any, prompt: Any, max_tokens: int, tokenizer: Any, temperature: float) -> Optional[Any]:
    if len(prompt.shape) == 1:
        prompt = prompt[None, :]
    if prompt.shape[1] == 0:
        return None

    end_sequence: List[int] = tokenizer.encode("</answer>")
    end_sequence_length: int = len(end_sequence)
    output: Any = mx.zeros((prompt.shape[1] + max_tokens,), dtype=mx.int32)
    output[:prompt.shape[1]] = prompt[0]
    current_length: int = prompt.shape[1]

    try:
        def sample(logits: Any) -> Any:
            if temperature > 0:
                logits /= temperature
            logprobs: Any = logits - mx.logsumexp(logits, keepdims=True)
            return mx.random.categorical(logprobs[None, :]).astype(mx.int32)[0]

        for _ in range(max_tokens):
            current_input: Any = output[:current_length][None, :]
            logits: Any = model(current_input)
            token_logits: Any = logits[0, -1]
            next_token: Any = sample(token_logits)
            token_value: int = next_token.item()
            output[current_length] = token_value
            current_length += 1

            if token_value == tokenizer.eos_token_id:
                break

            if current_length >= end_sequence_length:
                last_tokens: List[int] = output[current_length - end_sequence_length:current_length].tolist()
                if last_tokens == end_sequence:
                    break

        if current_length > prompt.shape[1]:
            return output[:current_length]

    except Exception as e:
        print(f"Generation error: {str(e)}")
        return None

    return None


def get_per_token_logps(model: Any, inputs: Any, lengths: Any) -> List[Any]:
    logits: Any = model(inputs).astype(mx.float16)
    logits = logits[:, :-1, :]
    targets: Any = inputs[:, 1:]
    per_token_logps: List[Any] = []
    for i in range(logits.shape[0]):
        seq_len: int = int(lengths[i]) - 1
        seq_logits: Any = logits[i, :seq_len]
        seq_targets: Any = targets[i, :seq_len]
        log_probs: Any = nn.log_softmax(seq_logits, axis=-1)
        token_log_probs: Any = mx.take_along_axis(
            log_probs,
            seq_targets.reshape(seq_len, 1),
            axis=-1
        ).squeeze(-1)
        per_token_logps.append(token_log_probs)
        mx.eval(logits)
    return per_token_logps


def grpo_loss(
        model: Any,
        ref_model: Optional[Any],
        tokenizer: Any,
        batch: Tuple[List[Any], List[Any], List[str], List[str]],
        reward_funcs: Optional[List[Callable[..., List[float]]]] = None,
        beta: float = 0.1,
        group_size: int = 4,
        epsilon: float = 1e-4,
        max_tokens: int = 64,
        temperature: float = 1.0
) -> Tuple[Any, int, Dict[str, Any]]:
    prompt_tokens, answer_tokens, prompt_text, answer_text = batch
    batch_size: int = len(prompt_tokens)

    # Generation logic remains the same
    all_completions: List[Any] = []
    all_completion_texts: List[str] = []

    for i in range(0, batch_size, batch_size):
        batch_prompts: List[Any] = prompt_tokens[i:i + batch_size]
        for prompt in batch_prompts:
            prompt_tensor: Any = mx.array(prompt)
            for _ in range(group_size):
                try:
                    completion_ids: Optional[Any] = generate_grpo(model, prompt_tensor, max_tokens, tokenizer,
                                                                  temperature)
                    if completion_ids is not None:
                        completion_text: str = tokenizer.decode(completion_ids.tolist())
                        all_completions.append(completion_ids)
                        all_completion_texts.append(completion_text)

                        # Clear completion tensors
                        mx.eval(completion_ids)
                        del completion_ids
                except Exception as e:
                    print(f"Generation error: {e}")
                    continue

        mx.metal.clear_cache()

    # Prepare inputs
    expanded_answers: List[str] = []
    expanded_prompts: List[str] = []
    for i in range(batch_size):
        expanded_answers.extend([answer_text[i]] * group_size)
        expanded_prompts.extend([prompt_text[i]] * group_size)

    max_length: int = max(ids.shape[0] for ids in all_completions)
    padded_completions: List[Any] = []
    attention_masks: List[Any] = []

    for completion_ids in all_completions:
        padding_length: int = max_length - completion_ids.shape[0]
        if padding_length > 0:
            padding: Any = mx.zeros((padding_length,), dtype=completion_ids.dtype)
            padded_ids: Any = mx.concatenate([completion_ids, padding])
            mask: Any = mx.concatenate([mx.ones_like(completion_ids), mx.zeros_like(padding)])
        else:
            padded_ids = completion_ids
            mask = mx.ones_like(completion_ids)
        padded_completions.append(padded_ids)
        attention_masks.append(mask)

    inputs: Any = mx.stack(padded_completions)
    attention_mask: Any = mx.stack(attention_masks)
    lengths: Any = attention_mask.sum(axis=1)

    # Current policy probabilities
    token_log_probs: List[Any] = get_per_token_logps(model, inputs, lengths)

    mx.eval(token_log_probs)
    mx.metal.clear_cache()

    # Reference policy probabilities
    if ref_model is None:
        ref_token_log_probs: List[Any] = token_log_probs
    else:
        ref_token_log_probs = get_per_token_logps(ref_model, inputs, lengths)

    max_len: int = max(x.shape[0] for x in token_log_probs)
    padded_log_probs: List[Any] = []
    padded_ref_log_probs: List[Any] = []

    for i in range(len(token_log_probs)):
        seq_len: int = token_log_probs[i].shape[0]
        padding: Any = mx.zeros((max_len - seq_len,))

        padded_log_probs.append(mx.concatenate([token_log_probs[i], padding]))
        padded_ref_log_probs.append(mx.concatenate([ref_token_log_probs[i], padding]))

    token_log_probs = mx.stack(padded_log_probs)
    ref_token_log_probs = mx.stack(padded_ref_log_probs)

    # Calculate rewards and advantages
    rewards: Any = mx.zeros((len(all_completions),))
    if reward_funcs is None:
        reward_funcs = []
    for reward_func in reward_funcs:
        func_rewards: Any = mx.array(reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers
        ))
        rewards += func_rewards

    if len(reward_funcs) > 1:
        rewards /= len(reward_funcs)

    # Reshape rewards and compute advantages following GRPO formula
    rewards_reshaped: Any = rewards.reshape(batch_size, group_size)
    mean_rewards: Any = mx.broadcast_to(mx.mean(rewards_reshaped, axis=1)[:, None],
                                        (rewards_reshaped.shape[0], group_size)).reshape(-1)
    std_rewards: Any = mx.broadcast_to(mx.std(rewards_reshaped, axis=1)[:, None],
                                       (rewards_reshaped.shape[0], group_size)).reshape(-1)
    advantages: Any = (rewards - mean_rewards) / (std_rewards + epsilon)

    # Compute KL divergence using Schulman's approximator
    kl_div: Any = mx.exp(token_log_probs - ref_token_log_probs) - (token_log_probs - ref_token_log_probs) - 1

    # Create mask for valid tokens
    length_mask: Any = mx.arange(inputs.shape[1] - 1)[None, :] < (lengths[:, None] - 1)

    # Compute policy ratio
    policy_ratio: Any = mx.exp(mx.array(token_log_probs - mx.stop_gradient(ref_token_log_probs)))

    # Compute per-token loss following GRPO formula
    per_token_loss: Any = -((policy_ratio * advantages.reshape(-1, 1) - beta * kl_div) * length_mask)

    # Average over tokens
    sequence_sums: Any = per_token_loss.sum(axis=1)
    sequence_lengths: Any = length_mask.sum(axis=1)
    loss: Any = (sequence_sums / sequence_lengths).mean()

    # Calculate mean KL divergence for metrics
    mean_kl: Any = ((kl_div * length_mask).sum(axis=1) / length_mask.sum(axis=1)).mean()

    # Collect reward metrics
    reward_metrics: Dict[str, Any] = {}
    for reward_func in reward_funcs:
        func_name: str = reward_func.__name__
        func_rewards: Any = mx.array(reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers
        ))
        reward_metrics[f'{func_name}_mean'] = mx.mean(func_rewards)
        reward_metrics[f'{func_name}_std'] = mx.std(func_rewards)

    metrics: Dict[str, Any] = {
        'total_rewards_mean': mx.mean(rewards),
        'total_rewards_std': mx.std(rewards),
        'grouped_rewards_mean': mx.mean(rewards_reshaped),
        'grouped_rewards_std': mx.std(rewards_reshaped),
        'kl': mean_kl,
        **reward_metrics
    }
    mx.metal.clear_cache()

    return loss, sequence_lengths.sum(), metrics


def iterate_grpo_batches(
        dataset: List[Tuple[Any, Any, str, str]],
        tokenizer: Any,
        batch_size: int,
        max_seq_length: int,
        train: bool = False
) -> Iterator[Tuple[List[Any], List[Any], List[str], List[str]]]:
    if not dataset or not isinstance(dataset[0], tuple) or len(dataset[0]) != 4:
        raise ValueError("Dataset must be list of (prompt_tokens, answer_tokens, prompt_str, answer_str) tuples")

    # Sort by length but use generator to avoid keeping full sorted list in memory
    def length_key(i: int) -> int:
        return len(dataset[i][0]) + len(dataset[i][1])

    idx: List[int] = sorted(range(len(dataset)), key=length_key)

    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )

    step: int = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    # Use generator for batch indices
    def batch_index_generator() -> Iterator[List[int]]:
        for i in range(0, len(idx) - batch_size + 1, batch_size):
            yield idx[i: i + batch_size: step]

    while True:
        indices: Iterator[List[int]]
        if train:
            indices = np.random.permutation(list(batch_index_generator()))
        else:
            indices = batch_index_generator()

        for batch_idx in indices:
            current_batch: List[Tuple[Any, Any, str, str]] = [dataset[j] for j in batch_idx]

            prompts_tokens: List[Any] = [item[0] for item in current_batch]
            answers_tokens: List[Any] = [item[1] for item in current_batch]
            prompts_text: List[str] = [item[2] for item in current_batch]
            answers_text: List[str] = [item[3] for item in current_batch]

            if any(len(p) > max_seq_length for p in prompts_tokens):
                print(
                    f"[WARNING] Some prompts are longer than {max_seq_length} tokens. "
                    "Long prompts will be truncated."
                )

            yield prompts_tokens, answers_tokens, prompts_text, answers_text

        if not train:
            break


def evaluate_grpo(
        model: Any,
        ref_model: Optional[Any],
        dataset: List[Tuple[Any, Any, str, str]],
        tokenizer: Any,
        batch_size: int,
        num_batches: int,
        beta: float,
        epsilon: float,
        group_size: int,
        max_seq_length: int,
        reward_funcs: Optional[List[Callable[..., List[float]]]] = None,
        loss_fn: Callable[..., Tuple[Any, int, Dict[str, Any]]] = grpo_loss,
        iterate_batches: Callable[
            ..., Iterator[Tuple[List[Any], List[Any], List[str], List[str]]]] = iterate_grpo_batches
) -> Tuple[float, int, Dict[str, float]]:
    """
    Evaluate model using GRPO loss.
    Returns:
        tuple: (average loss, number of tokens, average metrics)
    """
    all_losses: Any = 0
    ntokens: int = 0
    all_metrics: Optional[Dict[str, Any]] = None  # Initialize metrics dictionary

    # Create iterator for batches
    index_iterator: Iterator[int] = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    # Iterate through batches
    for _, batch in zip(
            index_iterator,
            iterate_batches(
                dataset=dataset,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
            ),
    ):
        # Calculate loss for current batch
        losses, toks, metrics = loss_fn(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            reward_funcs=reward_funcs,
            beta=beta,
            group_size=group_size,
            epsilon=epsilon,
            ref_model=ref_model
        )

        # Accumulate losses and tokens
        all_losses += losses * toks
        ntokens += toks

        # Accumulate metrics
        if all_metrics is None:
            all_metrics = {k: v * toks for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks

        # Evaluate accumulated values
        mx.eval(all_losses, ntokens)

    # Aggregate across distributed workers
    all_losses = mx.distributed.all_sum(all_losses, stream=mx.gpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    # Calculate averages
    avg_metrics: Dict[str, float] = {k: (v / ntokens).item() for k, v in all_metrics.items()}
    avg_loss: float = (all_losses / ntokens).item()

    return avg_loss, ntokens, avg_metrics


def train_grpo(
        model: Any,
        ref_model: Optional[Any],
        tokenizer: Any,
        optimizer: Any,
        train_dataset: List[Tuple[Any, Any, str, str]],
        val_dataset: List[Tuple[Any, Any, str, str]],
        reward_funcs: Optional[List[Callable[..., List[float]]]] = [
            r1_accuracy_reward_func,
            r1_int_reward_func,
            r1_strict_format_reward_func,
            r1_soft_format_reward_func,
            r1_count_xml
        ],
        args: GRPOTrainingArgs = GRPOTrainingArgs(),
        loss_fn: Callable[..., Tuple[Any, int, Dict[str, Any]]] = grpo_loss,
        iterate_batches: Callable[
            ..., Iterator[Tuple[List[Any], List[Any], List[str], List[str]]]] = iterate_grpo_batches,
        training_callback: Optional[TrainingCallback] = None,
) -> None:
    print(f"Starting GRPO training with {len(reward_funcs)} reward functions..., iters: {args.iters}")
    world: Any = mx.distributed.init()
    world_size: int = world.size()
    rank: int = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state: List[Any] = [model.state, optimizer.state]

    def step(batch: Tuple[List[Any], List[Any], List[str], List[str]]) -> Tuple[Any, int, Dict[str, Any]]:
        # Forward and backward pass
        (loss, toks, metrics), grad = loss_value_and_grad(
            model,
            tokenizer=tokenizer,
            batch=batch,
            reward_funcs=reward_funcs,
            beta=args.beta,
            group_size=args.group_size,
            epsilon=args.epsilon,
            ref_model=ref_model,
            max_tokens=args.max_completion_length,
        )

        # All reduce the gradients if running in distributed mode
        grad = average_gradients(grad)

        # Model update
        optimizer.update(model, grad)

        return loss, toks, metrics

    loss_value_and_grad: Callable[..., Tuple[Tuple[Any, int, Dict[str, Any]], Any]] = nn.value_and_grad(model, loss_fn)

    losses: Any = 0
    n_tokens: int = 0
    steps: int = 0
    trained_tokens: int = 0
    accumulated_metrics: Dict[str, Any] = {
        'total_rewards_mean': 0,
        'total_rewards_std': 0,
        'grouped_rewards_mean': 0,
        'grouped_rewards_std': 0,
        'kl': 0
    }
    for reward_func in reward_funcs:
        func_name: str = reward_func.__name__
        accumulated_metrics[f'{func_name}_mean'] = 0
        accumulated_metrics[f'{func_name}_std'] = 0

    start: float = time.perf_counter()
    for it, batch in zip(
            range(1, args.iters + 1),
            iterate_batches(
                dataset=train_dataset,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                train=True,
            ),
    ):
        # Report validation loss if needed, the first validation loss
        # is always measured before any training.
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop: float = time.perf_counter()
            val_loss, val_ntokens, val_metrics = evaluate_grpo(
                model=model,
                dataset=val_dataset,
                loss_fn=loss_fn,
                ref_model=ref_model,
                reward_funcs=reward_funcs,
                tokenizer=tokenizer,
                group_size=args.group_size,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                beta=args.beta,
                epsilon=args.epsilon,
                iterate_batches=iterate_batches,
            )
            val_time: float = time.perf_counter() - stop
            if rank == 0:
                val_metrics_str: str = (
                    f"Val loss {val_loss:.8f}, "
                    f"Val total_rewards_mean {val_metrics['total_rewards_mean']:.3f}, "
                    f"Val total_rewards_std {val_metrics['total_rewards_std']:.3f}, "
                    f"Val grouped_rewards_mean {val_metrics['grouped_rewards_mean']:.3f}, "
                    f"Val grouped_rewards_std {val_metrics['grouped_rewards_std']:.3f}, "
                    f"Val kl {val_metrics['kl']:.3f}"
                )

                # Add reward function specific metrics
                for reward_func in reward_funcs:
                    val_metrics_str += (
                        f", Val {reward_func.__name__}_mean {val_metrics[f'{reward_func.__name__}_mean']:.3f}, "
                        f"Val {reward_func.__name__}_std {val_metrics[f'{reward_func.__name__}_std']:.3f}"
                    )

                print(
                    f"Iter {it}: {val_metrics_str}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                training_callback.on_val_loss_report({
                    "iteration": it,
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    "val_time": val_time,
                })

            start = time.perf_counter()

        loss, toks, metrics = step(batch)
        losses += loss
        n_tokens += toks
        steps += 1

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, n_tokens)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss: float = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * mx.distributed.init().size()
            avg_metrics: Dict[str, Any] = {k: v / (steps * world_size) for k, v in accumulated_metrics.items()}
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate: float = optimizer.learning_rate.item()
            it_sec: float = args.steps_per_report / (stop - start)
            tokens_sec: float = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            peak_mem: float = mx.metal.get_peak_memory() / 1e9

            if rank == 0:
                train_metrics_str: str = (
                    f"Train loss {train_loss:.8f}, "
                    f"Total rewards mean {avg_metrics['total_rewards_mean']:.3f}, "
                    f"Total rewards std {avg_metrics['total_rewards_std']:.3f}, "
                    f"Grouped rewards mean {avg_metrics['grouped_rewards_mean']:.3f}, "
                    f"Grouped rewards std {avg_metrics['grouped_rewards_std']:.3f}, "
                    f"KL {avg_metrics['kl']:.3f}"
                )

                # Add reward function specific metrics
                for reward_func in reward_funcs:
                    func_name: str = reward_func.__name__
                    train_metrics_str += (
                        f", {func_name} mean {avg_metrics[f'{func_name}_mean']:.3f}, "
                        f"{func_name} std {avg_metrics[f'{func_name}_std']:.3f}"
                    )

                print(
                    f"Iter {it}: {train_metrics_str}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            if training_callback is not None:
                training_callback.on_train_loss_report({
                    "iteration": it,
                    "train_loss": train_loss,
                    **{f"train_{k}": v for k, v in avg_metrics.items()},
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                })

            losses = 0
            n_tokens = 0
            steps = 0
            start = time.perf_counter()

        # Save adapter weights
        if it % args.steps_per_save == 0:
            adapter_weights: Dict[str, Any] = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint: Path = Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    # Save final weights
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.")
