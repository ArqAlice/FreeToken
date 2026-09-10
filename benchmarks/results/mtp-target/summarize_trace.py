"""Summarize decode kernels after the final prefill expert kernel in a trace."""

import collections
import json
from pathlib import Path

trace = json.loads(Path('/tmp/mtp-target-trace.json').read_text())
events = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']
cutoff = max(e['ts'] + e['dur'] for e in events if '_prefill_nvfp4_moe_kernel' in e['name'])
totals = collections.defaultdict(lambda: dict(calls=0, microseconds=0))
for event in events:
    if event['ts'] < cutoff:
        continue
    name = event['name']
    category = ('expert_copy' if 'fast_index_copy_multi' in name else
                'bf16_gemv' if 'gemvx' in name else
                'expert_gemv' if '_decode_nvfp4_marlin_kernel' in name else
                'routing_cache' if '_lru_ensure_kernel' in name else 'other')
    totals[category]['calls'] += 1
    totals[category]['microseconds'] += event['dur']
report = dict(cutoff=cutoff, categories=totals,
              note='Excludes prefill up to its last expert kernel. Includes decode target and draft. '
                   'Kernel durations overlap across streams and are not exclusive wall time.')
Path('/tmp/mtp-target-kernels.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
