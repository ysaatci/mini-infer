"""What int8 weights cost in quality, and what they save in memory.

- WikiText-2 test perplexity: the standard language-modeling benchmark, so numbers compare with
  published ones. Non-overlapping 1024-token windows.
- Greedy agreement: bf16 answers the chat prompts; then both models read prompt + that answer and we
  count how often they pick the same next token. Reading the same text (not each its own answer) keeps
  one early difference from snowballing into a different answer.

python -m bench.quality --out bench/results/step8-quality.json
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, GenerationConfig

from bench.report import markdown_table, save_json
from mini_infer.engine import LLMEngine
from mini_infer.loader import load_model, resolve_model_dir
from mini_infer.quant import quantize_model

PROMPTS = json.loads((Path(__file__).parent / "chat_prompts.json").read_text())
WINDOW = 1024


@dataclass(frozen=True)
class QualityMetrics:
    wikitext2_ppl: float
    greedy_agreement: float  # share of answer positions where the top token matches bf16's
    weights_gb: float  # GPU memory held by the model after loading


def wikitext2_ids(tokenizer) -> torch.Tensor:
    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
    return tokenizer(text, return_tensors="pt").input_ids[0]


@torch.inference_mode()
def perplexity(model, ids: torch.Tensor) -> float:
    nll, count = 0.0, 0
    for start in range(0, len(ids) - 1, WINDOW):
        window = ids[start : start + WINDOW + 1].cuda()
        logits = model(window[None, :-1])[0]
        nll += F.cross_entropy(logits.float(), window[1:], reduction="sum").item()
        count += len(window) - 1
    return math.exp(nll / count)


@torch.inference_mode()
def top_tokens(model, sequences: list[tuple[list[int], list[int]]]) -> list[torch.Tensor]:
    """For each (prompt, answer): the model's top next-token choice at every answer position."""
    choices = []
    for prompt, answer in sequences:
        ids = torch.tensor([prompt + answer], device="cuda")
        logits = model(ids)[0, len(prompt) - 1 : -1]  # position i predicts token i + 1
        choices.append(logits.argmax(-1).cpu())
    return choices


def greedy_answers(model, prompts: list[list[int]], stop_ids: frozenset[int]) -> list[list[int]]:
    engine = LLMEngine(model, num_blocks=1000, max_batch_size=len(prompts), use_cuda_graphs=False)
    for i, ids in enumerate(prompts):
        engine.add_request(ids, 128, stop_ids=stop_ids, request_id=str(i))
    answers = {str(i): [] for i in range(len(prompts))}
    while engine.has_unfinished():
        for out in engine.step():
            answers[out.request_id].append(out.token)
    return [answers[str(i)] for i in range(len(prompts))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    eos = GenerationConfig.from_pretrained(model_dir).eos_token_id
    stop_ids = frozenset([eos] if isinstance(eos, int) else eos)
    wikitext = wikitext2_ids(tokenizer)
    prompts = [
        tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False),
                  add_special_tokens=False).input_ids
        for p in PROMPTS
    ]

    model = load_model(args.model)
    bf16_memory = torch.cuda.memory_allocated()
    answers = greedy_answers(model, prompts, stop_ids)
    sequences = list(zip(prompts, answers))
    reference = top_tokens(model, sequences)
    rows = [("bf16", "quality", QualityMetrics(perplexity(model, wikitext), 1.0, bf16_memory / 1e9))]

    quantize_model(model)  # in place: no second copy on the GPU
    int8_memory = torch.cuda.memory_allocated()
    choices = top_tokens(model, sequences)
    agreement = sum((a == b).sum().item() for a, b in zip(choices, reference)) / sum(len(a) for a in reference)
    rows.append(("int8", "quality", QualityMetrics(perplexity(model, wikitext), agreement, int8_memory / 1e9)))

    print(markdown_table(rows))
    save_json(args.out, vars(args) | {"out": str(args.out), "wikitext2_tokens": len(wikitext)}, rows)


if __name__ == "__main__":
    main()
