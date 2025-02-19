# Copyright © 2024 Apple Inc.

"""
Run with:

```
/path/to/mpirun \
 -np 2 \
 --hostfile /path/to/hosts.txt \
 python /path/to/pipeline_generate.py --prompt "hello world"
```

Make sure you can run MLX over MPI on two hosts. For more information see the
documentation:

https://ml-explore.github.io/mlx/build/html/usage/distributed.html).
"""

import argparse

import mlx.core as mx

from src.llms.mlx_lm.utils import load, stream_generate

parser = argparse.ArgumentParser(description="LLM pipelined inference example")
parser.add_argument(
    "--model",
    default="mlx-community/DeepSeek-R1-3bit",
    help="HF repo or path to local model.",
)
parser.add_argument(
    "--prompt",
    "-p",
    default="Write a quicksort in C++.",
    help="Message to be processed by the model ('-' reads from stdin)",
)
parser.add_argument(
    "--max-tokens",
    "-m",
    type=int,
    default=256,
    help="Maximum number of tokens to generate",
)
args = parser.parse_args()

model, tokenizer = load(args.model, lazy=True)

messages = [{"role": "user", "content": args.prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

group = mx.distributed.init()
rank = group.rank()
model.model.pipeline(group)
mx.eval(model.parameters())

# Synchronize processes before generation to avoid timeout if downloading
# model for the first time.
mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))


def rprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


for response in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens):
    rprint(response.text, end="", flush=True)

rprint()
rprint("=" * 10)
rprint(
    f"Prompt: {response.prompt_tokens} tokens, "
    f"{response.prompt_tps:.3f} tokens-per-sec"
)
rprint(
    f"Generation: {response.generation_tokens} tokens, "
    f"{response.generation_tps:.3f} tokens-per-sec"
)
rprint(f"Peak memory: {response.peak_memory:.3f} GB")
