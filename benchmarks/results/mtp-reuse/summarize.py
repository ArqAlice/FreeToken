"""Summarize completed process comparisons without mixing prompts or depths."""
import json
from pathlib import Path
import statistics

root = Path('/tmp')
profiles = {}
summary = []
for profile in ('baseline', 'expert', 'gdn', 'combined'):
    path = root / f'mtp-reuse-{profile}.json'
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    profiles[profile] = data
    for mode in (0, 1, 3):
        rows = [r for r in data['runs'] if r['mtp'] == mode]
        if not rows:
            continue
        item = dict(profile=profile, mtp=mode, repeats=len(rows),
                    median_tps=statistics.median(r['decode_tokens_per_second'] for r in rows),
                    min_tps=min(r['decode_tokens_per_second'] for r in rows),
                    max_tps=max(r['decode_tokens_per_second'] for r in rows),
                    median_ttft=statistics.median(r['ttft_seconds'] for r in rows),
                    total_misses=sum(r['expert_stats']['misses'] for r in rows),
                    total_estimated_fetch_bytes=sum(r['expert_stats']['estimated_fetch_bytes'] for r in rows),
                    total_proposed=sum(r['proposed'] for r in rows),
                    total_accepted=sum(r['accepted'] for r in rows),
                    cuda_allocated_bytes=max(r['cuda_allocated_bytes'] for r in rows))
        reference = profiles.get('baseline', {}).get('runs', [])
        pairs = [(r, next((b for b in reference if b['mtp'] == mode and b['repeat'] == r['repeat']), None)) for r in rows]
        item['token_pairs_equal'] = [r['token_ids'] == b['token_ids'] for r, b in pairs if b]
        summary.append(item)
print(json.dumps(summary, indent=2))
(root / 'mtp-reuse-summary.json').write_text(json.dumps(summary, indent=2))
