"""Isolate each projection/cache setting in a fresh process and CUDA graphs."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = '/opt/freetoken/benchmarks'
model = '/models/huggingface/hub/models--aday777--Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP/snapshots/18bd7d69491d4709ac1933e95343e15f045ea21d'
base = {**os.environ, 'PYTHONPATH': '/opt/freetoken/python',
        'FREETOKEN_MTP_BATCHED_LINEAR': '0', 'FREETOKEN_MTP_PARALLEL_LINEAR': '0'}

def run(name, script, args, half=0, distance=0, shared=0):
    env = {**base, 'FREETOKEN_EXPERT_LRFU_HALF_LIFE': str(half),
           'FREETOKEN_EXPERT_LAYER_DISTANCE': str(distance),
           'FREETOKEN_GDN_SHARED_INPUT': str(shared)}
    output = f'/tmp/mtp-reuse-{name}.json'
    command = [sys.executable, f'{root}/{script}', '--model', model, '--output', output, *args]
    print(json.dumps(dict(start=name, command=command, settings={k:v for k,v in env.items() if k.startswith('FREETOKEN_')})), flush=True)
    with open(f'/tmp/mtp-reuse-{name}.log', 'w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(Path(output).read_text())
    if name == 'validation':
        assert all(c['equal'] for c in result['checks'].values()), result['checks'].keys()
        assert all(c['equal'] for c in result['contexts'])
    print(json.dumps(dict(completed=name)), flush=True)

run('validation', 'bench_mtp.py', [
    '--speculative-tokens', '3', '--tokens', '128', '--warmup-tokens', '64',
    '--repeats', '1', '--validate', '--sampling-smoke', '--sampling-tokens', '64',
    '--max-extend-tokens', '8192', '--max-seq-len', '1000128', '--pages', '15629',
    '--moe-slots', '3073', '--validation-contexts', '8190'], 256, 2, 1)

for name, half, distance, shared in [('baseline', 0, 0, 0), ('expert', 256, 2, 0),
                                     ('gdn', 0, 0, 1), ('combined', 256, 2, 1)]:
    run(name, 'bench_mtp_chat.py', [
        '--modes', '0,1,3', '--tokens', '256', '--warmup-tokens', '128', '--repeats', '3',
        '--reset-cache-between-runs', '--record-expert-stats'], half, distance, shared)
