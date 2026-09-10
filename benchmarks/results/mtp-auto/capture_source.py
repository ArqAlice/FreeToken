"""Record runtime delta against the pre-change source snapshot in the dev container."""

import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path


root = Path('/opt/freetoken')
before = Path('/tmp/mtp-auto-before-python')
after = root / 'python'
assert before.is_dir()
paths = sorted({p.relative_to(before) for p in before.rglob('*.py')} |
               {p.relative_to(after) for p in after.rglob('*.py')})
patch, changed = [], []
for relative in paths:
    old, new = before / relative, after / relative
    a = old.read_text() if old.exists() else ''
    b = new.read_text() if new.exists() else ''
    if a == b:
        continue
    name = 'python/' + relative.as_posix()
    changed.append(name)
    patch.extend(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                    fromfile='a/' + name if old.exists() else '/dev/null',
                                    tofile='b/' + name if new.exists() else '/dev/null'))
Path('/tmp/mtp-runtime-change.patch').write_text(''.join(patch))
tracked = changed + [
    'tests/scheduler/test_mtp.py', 'tests/scheduler/test_cache_rebuild.py',
    'tests/engine/test_kv_quant_config.py', 'tests/engine/test_speculative.py',
    'tests/models/qwen4_exp/test_ple_disk.py', 'tests/models/qwen4_exp/test_skeleton.py',
    'benchmarks/bench_mtp.py', 'benchmarks/bench_mtp_chat.py',
    'benchmarks/bench_mtp_http.py', 'benchmarks/bench_nvfp4_moe.py',
]
report = dict(
    branch='feat/mtp-support', base_head='3381015664c3acc5003b04fcad9a29742429792c',
    patch_reference='Pre-turn runtime snapshot, including earlier uncommitted MTP work; not HEAD',
    checkpoint='aday777/Qwen3.8-Flash-Next-Uncensored-NVFP4-MTP',
    revision='18bd7d69491d4709ac1933e95343e15f045ea21d',
    versions={name: importlib.metadata.version(name) for name in ('torch', 'triton', 'transformers')},
    geometry=dict(max_seq_len=1000128, pages=15629, moe_slots=3073, request_slots=2,
                  max_prefill_length=8192, kv_dtype='nvfp4', cache_type='naive'),
    runtime_changed=changed,
    sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in tracked},
)
Path('/tmp/mtp-source.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(changed=changed, versions=report['versions']), indent=2))
