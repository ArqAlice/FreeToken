"""Record the runtime delta against the local pre-reuse snapshot."""
import difflib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

root = Path('/opt/freetoken')
before = Path('/tmp/mtp-reuse-before-python')
after = root / 'python'
assert before.is_dir()
changed, patch = [], []
for relative in sorted({p.relative_to(before) for p in before.rglob('*.py')} |
                       {p.relative_to(after) for p in after.rglob('*.py')}):
    old, new = before / relative, after / relative
    a, b = old.read_text() if old.exists() else '', new.read_text() if new.exists() else ''
    if a == b:
        continue
    name = 'python/' + relative.as_posix()
    changed.append(name)
    patch.extend(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                    fromfile='a/' + name if old.exists() else '/dev/null',
                                    tofile='b/' + name if new.exists() else '/dev/null'))
Path('/tmp/mtp-reuse-runtime.patch').write_text(''.join(patch))
tracked = changed + ['tests/layers/test_linear_mtp.py', 'tests/moe/test_offload.py',
                     'tests/moe/test_fused_copy.py', 'tests/scheduler/test_cache_rebuild.py',
                     'tests/scheduler/test_mtp.py', 'benchmarks/bench_mtp.py',
                     'benchmarks/bench_mtp_chat.py']
tracked += [p.relative_to(root).as_posix() for p in (root / 'benchmarks/results/mtp-reuse').glob('*.py')]
report = dict(branch='feat/mtp-support', head='3381015664c3acc5003b04fcad9a29742429792c',
              reference='Local branch before weight reuse; includes earlier uncommitted MTP changes',
              python=platform.python_version(),
              versions={name: importlib.metadata.version(name) for name in ('torch', 'triton', 'transformers')},
              runtime_changed=changed,
              sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in tracked})
Path('/tmp/mtp-reuse-source.json').write_text(json.dumps(report, indent=2))
print(json.dumps(dict(changed=changed, versions=report['versions']), indent=2))
