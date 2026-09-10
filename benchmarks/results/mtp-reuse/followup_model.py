"""Check the new projection's distribution delta and automatic depth transitions."""
import json
import os
from pathlib import Path
import subprocess
import sys

model = '/models/huggingface/hub/models--aday777--Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP/snapshots/18bd7d69491d4709ac1933e95343e15f045ea21d'
env = {**os.environ, 'PYTHONPATH': '/opt/freetoken/python',
       'FREETOKEN_GDN_SHARED_INPUT': '1', 'FREETOKEN_EXPERT_LRFU_HALF_LIFE': '256',
       'FREETOKEN_EXPERT_LAYER_DISTANCE': '2', 'FREETOKEN_MTP_BATCHED_LINEAR': '0',
       'FREETOKEN_MTP_PARALLEL_LINEAR': '0'}
subprocess.run([sys.executable, '/opt/freetoken/benchmarks/results/mtp-reuse/probe_cold.py'],
               env=env, check=True)
jobs = [
    ('numerics', 'bench_mtp_chat.py', ['--modes', '3', '--tokens', '128',
     '--warmup-tokens', '64', '--repeats', '0', '--compare-gdn-input']),
    ('auto', 'bench_mtp.py', ['--mtp-auto', '--force-auto-depths', '--speculative-tokens', '3',
     '--tokens', '128', '--warmup-tokens', '64', '--repeats', '1', '--validate',
     '--max-extend-tokens', '8192', '--max-seq-len', '1000128', '--pages', '15629',
     '--moe-slots', '3073']),
]
for name, script, args in jobs:
    command = [sys.executable, f'/opt/freetoken/benchmarks/{script}', '--model', model,
               '--output', f'/tmp/mtp-reuse-{name}.json', *args]
    print(json.dumps(dict(start=name, command=command)), flush=True)
    with open(f'/tmp/mtp-reuse-{name}.log', 'w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps(dict(completed=name)), flush=True)
