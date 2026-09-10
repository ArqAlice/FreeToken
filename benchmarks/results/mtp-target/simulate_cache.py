"""Replay recorded expert routes without model execution or timing claims."""

import argparse
import json
from pathlib import Path

import numpy as np


def simulate(trace, policy, half_life=1024):
    slots = np.array(trace['initial']['ids'])
    usage = np.array(trace['initial']['usage'])
    size = trace['experts_per_layer']
    total = (max(r['layer'] for r in trace['routes']) + 1) * size
    mapping = np.full(total, -1, dtype=np.int64)
    occupied = slots >= 0
    mapping[slots[occupied]] = np.arange(len(slots))[occupied]
    last = np.zeros(total, dtype=np.int64)
    last[slots[occupied]] = usage[occupied]
    previous = np.zeros(total, dtype=np.int64)
    frequency = np.zeros(total)
    counts = dict(active=0, misses=0, decode_active=0, decode_misses=0)
    start = int(usage.max())
    for step, row in enumerate(trace['routes'], start + 1):
        ids = np.unique(np.array(row['ids']).reshape(-1) + row['layer'] * size)
        if policy == 'lrfu':
            frequency[ids] = frequency[ids] * np.exp2(-(step - last[ids]) / half_life) + 1
        else:
            frequency[ids] += 1
        previous[ids], last[ids] = last[ids], step
        hit = mapping[ids]
        missing = ids[hit < 0]
        usage[hit[hit >= 0]] = step
        if policy in ('lru', 'speculative_lru'):
            score = usage.astype(float)
        elif policy == 'lru2':
            score = previous[np.maximum(slots, 0)].astype(float)
        elif policy == 'lfu':
            score = frequency[np.maximum(slots, 0)]
        else:
            safe = np.maximum(slots, 0)
            score = frequency[safe] * np.exp2(-(step - last[safe]) / half_life)
        score[slots < 0] = -np.inf
        score[hit[hit >= 0]] = np.inf
        victims = np.lexsort((np.arange(len(slots)), usage, score))[:len(missing)]
        old = slots[victims]
        mapping[old[old >= 0]] = -1
        slots[victims] = missing
        mapping[missing] = victims
        usage[victims] = step
        if policy == 'speculative_lru' and len(row['ids']) > 1:
            certain = np.array(row['ids'][0]) + row['layer'] * size
            tentative = np.setdiff1d(ids, certain)
            usage[mapping[tentative]] = step - half_life
        counts['active'] += len(ids)
        counts['misses'] += len(missing)
        if row['phase'] == 'decode':
            counts['decode_active'] += len(ids)
            counts['decode_misses'] += len(missing)
    return dict(policy=policy, half_life=half_life if policy == 'lrfu' else None,
                speculative_penalty=half_life if policy == 'speculative_lru' else None, **counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    trace = data['diagnostics']['1']['routing_trace']
    result = [simulate(trace, p) for p in ('lru', 'lru2', 'lfu')]
    result += [simulate(trace, 'lrfu', h) for h in (256, 512, 1024, 2048, 4096)]
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
