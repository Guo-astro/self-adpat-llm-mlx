import math
import time
from functools import partial
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

# New imports for Wandb, system monitoring, and Rich
import wandb
import psutil
import pynvml
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from src.transformer_lm import datasets
from src.transformer_lm.Linear import TransformerLM


def to_samples(context_size, dataset):
    tokens = dataset.size
    window_size = context_size + 1  # include target token
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]


def iterate_batches(batch_size, context_size, dataset):
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s: s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0


def main(args):
    console = Console()

    # Initialize Wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
    )
    config = wandb.config

    batch_size = args.batch_size
    context_size = args.context_size
    steps_per_eval = args.steps_per_eval
    steps_per_report = args.steps_per_report

    # Load vocab and dataset.
    vocab, train, valid, test = datasets.load_dataset(args.dataset)

    # Initialize model using our SVD-based Transformer.
    model = TransformerLM(
        len(vocab), args.num_blocks, args.dim, args.num_heads, args.checkpoint
    )
    mx.eval(model.parameters())
    nparams = sum(x.size for _, x in tree_flatten(model.parameters()) if "embedding" not in _)

    header = Panel.fit(
        f"Training a Transformer with [bold magenta]{nparams / 1024 ** 2:.3f} M[/bold magenta] Parameters",
        title="Model Information",
        border_style="green",
    )
    console.print(header)

    def loss_fn(model, x, y, reduce=True):
        logits = model(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))

    optimizer = optim.AdamW(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )

    def eval_fn(dataset):
        inputs, targets = map(mx.array, to_samples(context_size, dataset))
        loss = 0
        for s in range(0, targets.shape[0], batch_size):
            bx, by = inputs[s: s + batch_size], targets[s: s + batch_size]
            bx, by = map(mx.array, (bx, by))
            losses = loss_fn(model, bx, by, reduce=False)
            loss += mx.sum(losses).item()
        return loss / targets.size

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        return loss

    train_iterator = iterate_batches(batch_size, context_size, train)
    losses = []
    tic = time.perf_counter()

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )
    task = progress.add_task("[green]Training...", total=args.num_iters)

    with progress:
        for it, (inputs, targets) in zip(range(args.num_iters), train_iterator):
            inputs, targets = map(mx.array, (inputs, targets))
            optimizer.learning_rate = min(1, it / args.lr_warmup) * args.learning_rate
            loss = step(inputs, targets)
            mx.eval(state)
            losses.append(loss.item())
            progress.update(task, advance=1)

            if (it + 1) % steps_per_report == 0:
                train_loss = np.mean(losses)
                toc = time.perf_counter()
                it_per_sec = steps_per_report / (toc - tic)

                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                memory_used = memory_info.used / (1024 ** 3)
                memory_total = memory_info.total / (1024 ** 3)

                wandb.log({
                    "train_loss": train_loss,
                    "it_per_sec": it_per_sec,
                    "cpu_usage_percent": cpu_percent,
                    "memory_used_GB": memory_used,
                    "memory_total_GB": memory_total,
                    "iteration": it + 1
                })

                table = Table(title=f"Iteration {it + 1}", border_style="blue")
                table.add_column("Metric", style="cyan", no_wrap=True)
                table.add_column("Value", style="magenta")
                table.add_row("Train Loss", f"{train_loss:.4f}")
                table.add_row("Iterations/sec", f"{it_per_sec:.2f}")
                table.add_row("CPU Usage", f"{cpu_percent}%")
                table.add_row("Memory Used (GB)", f"{memory_used:.2f} / {memory_total:.2f}")
                console.print(table)

                losses = []
                tic = time.perf_counter()

            if (it + 1) % steps_per_eval == 0:
                val_loss = eval_fn(valid)
                toc = time.perf_counter()
                val_ppl = math.exp(val_loss)
                eval_time = toc - tic

                wandb.log({
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "val_time_sec": eval_time,
                    "iteration": it + 1
                })

                val_panel = Panel.fit(
                    f"Validation at Iteration {it + 1}\n"
                    f"• [bold]Val Loss[/bold]: {val_loss:.4f}\n"
                    f"• [bold]Val PPL[/bold]: {val_ppl:.2f}\n"
                    f"• [bold]Time Taken[/bold]: {eval_time:.2f} seconds",
                    title="Validation Results",
                    border_style="yellow",
                )
                console.print(val_panel)
                tic = time.perf_counter()

    if args.eval_test:
        test_loss = eval_fn(test)
        test_ppl = math.exp(test_loss)
        test_panel = Panel.fit(
            f"Test Results\n"
            f"• [bold]Test Loss[/bold]: {test_loss:.4f}\n"
            f"• [bold]Test PPL[/bold]: {test_ppl:.2f}",
            title="Test Evaluation",
            border_style="red",
        )
        console.print(test_panel)
        wandb.log({
            "test_loss": test_loss,
            "test_ppl": test_ppl
        })

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a decoder-only Transformer LM with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the GPU for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the RNGs.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ptb",
        choices=["enwik8", "ptb", "wikitext2", "wikitext103"],
        help="Dataset to train and evaluate on.",
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=1024,
        help="Context size in tokens of the model.",
    )
    parser.add_argument(
        "--num_blocks", type=int, default=12, help="Number of Transformer blocks."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="Dimensionality of embeddings and hidden layers.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=16,
        help="Number of heads used for multi-head attention",
    )
    parser.add_argument(
        "--checkpoint", action="store_true", help="Perform gradient checkpointing"
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Minibatch size.")
    parser.add_argument(
        "--num_iters", type=int, default=100000, help="Iterations to train for."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="AdamW learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Set the weight decay"
    )
    parser.add_argument(
        "--lr_warmup", type=int, default=200, help="LR linear warmup iterations"
    )
    parser.add_argument(
        "--steps_per_report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps_per_eval",
        type=int,
        default=1000,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--eval_test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="transformer-training",
        help="Wandb project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name.",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    mx.set_default_device(mx.gpu)
    main(args)
