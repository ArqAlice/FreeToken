"""Measure speculative correction with the checkpoint's sampling filters."""

import argparse
import json
import time
from pathlib import Path

import torch

from freetoken.engine.sample import BatchSamplingArgs, Sampler
from freetoken.engine.speculative import acceptance_flags, acceptance_prefix, matching_prefix, residual_probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(991)
    device = torch.device("cuda")
    sampler = Sampler(device, 248320)
    params = BatchSamplingArgs(torch.ones(1, device=device),
                               torch.tensor([20], dtype=torch.int32, device=device),
                               torch.tensor([.95], device=device))
    logits = torch.randn((4, sampler.vocab_size), device=device) * 4
    target = sampler.probabilities(logits, params)
    draft = sampler.probabilities(logits[:3] + .25 * torch.randn_like(logits[:3]), params)
    output = {"vocab": sampler.vocab_size, "draft_tokens": 3,
              "temperature": 1., "top_k": 20, "top_p": .95, "cases": []}
    for first_rejection in (0, 1, 3):
        tokens = (target[:3] * draft).argmax(-1)
        uniforms = torch.zeros(3, device=device)
        if first_rejection < 3:
            tokens[first_rejection] = (draft[first_rejection] - target[first_rejection]).argmax()
            uniforms[first_rejection] = 1.

        def before():
            flags = acceptance_flags(target[:3], draft, tokens, uniforms=uniforms)
            accepted = int(flags.int().cumprod(0).sum().item())
            distribution = (target[-1:] if accepted == 3 else
                            residual_probabilities(target[accepted:accepted + 1], draft[accepted:accepted + 1]))
            sampler.sample_probs(distribution)
            return accepted

        def after():
            accepted = acceptance_prefix(target[:3], draft, tokens, uniforms=uniforms)
            sampler.sample_speculative(target, draft, accepted)
            return int(accepted.item())

        result = {"accepted": first_rejection}
        for name, fn in (("before", before), ("after", after)):
            for _ in range(10):
                assert fn() == first_rejection
            samples = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(100):
                    fn()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1e6 / 100)
            result[name + "_us"] = sorted(samples)[len(samples) // 2]
        output["cases"].append(result)
    output["proposal_sampling"] = []
    draft_logits = logits[:3] + .25 * torch.randn_like(logits[:3])
    for count in (1, 3):
        def stochastic():
            distributions = [sampler.probabilities(row, params) for row in draft_logits[:count].split(1)]
            proposals = torch.cat([sampler.sample_probs(q) for q in distributions])
            q = torch.cat(distributions)
            p = sampler.probabilities(logits[:count + 1], params)
            accepted = acceptance_prefix(p[:-1], q, proposals)
            sampler.sample_speculative(p, q, accepted)
            return int(accepted.item())

        def greedy():
            proposals = torch.cat([row.argmax(-1) for row in draft_logits[:count].split(1)])
            p = sampler.probabilities(logits[:count + 1], params)
            verified = sampler.sample_probs(p)
            return int(matching_prefix(verified, proposals).item())

        row = {"draft_tokens": count}
        for name, fn in (("stochastic", stochastic), ("greedy", greedy)):
            for _ in range(10):
                fn()
            samples = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(100):
                    fn()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1e6 / 100)
            row[name + "_us"] = sorted(samples)[len(samples) // 2]
        output["proposal_sampling"].append(row)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
