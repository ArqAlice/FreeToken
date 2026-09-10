"""Screen layer-distance eviction on a previously recorded routing sequence."""
import json
from pathlib import Path
import numpy as np

trace = json.loads(Path('/opt/freetoken/benchmarks/results/mtp-target/routes.json').read_text())['diagnostics']['1']['routing_trace']
size = trace['experts_per_layer']
layers = max(r['layer'] for r in trace['routes']) + 1
routes = [(r['layer'], np.unique(np.array(r['ids']).reshape(-1) + r['layer']*size)) for r in trace['routes']]
results = []
for half, penalty in ((0, 0), (256, 0), (0, 1), (0, 2), (256, .5), (256, 1), (256, 2), (256, 4), (256, 8)):
    slots = np.array(trace['initial']['ids'])
    usage = np.array(trace['initial']['usage'])
    mapping = np.full(layers*size, -1, dtype=np.int64)
    occupied = slots >= 0
    mapping[slots[occupied]] = np.arange(len(slots))[occupied]
    last = np.zeros(layers*size, dtype=np.int64)
    last[slots[occupied]] = usage[occupied]
    frequency = np.zeros(layers*size)
    misses = 0
    for step, (layer, ids) in enumerate(routes, int(usage.max())+1):
        if half:
            frequency[ids] = frequency[ids]*np.exp2(-(step-last[ids])/half)+1
        last[ids] = step
        hit = mapping[ids]
        missing = ids[hit < 0]
        usage[hit[hit >= 0]] = step
        safe = np.maximum(slots, 0)
        score = (last[safe] + half*np.log2(np.maximum(frequency[safe], 1e-30))
                 if half else usage.astype(float))
        score -= penalty*((safe//size-layer-1) % layers)
        score[slots < 0] = -np.inf
        score[hit[hit >= 0]] = np.inf
        victims = np.lexsort((np.arange(len(slots)), usage, score))[:len(missing)]
        old = slots[victims]
        mapping[old[old >= 0]] = -1
        slots[victims] = missing
        mapping[missing] = victims
        usage[victims] = step
        misses += len(missing)
    result = dict(half_life=half, distance_penalty=penalty, misses=misses)
    results.append(result)
    print(json.dumps(result), flush=True)
Path('/tmp/mtp-reuse-cache-probe.json').write_text(json.dumps(results, indent=2))
