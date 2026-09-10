"""Record the additional runtime delta and the sources used in its verification."""

import difflib
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


root = Path('/opt/freetoken')
before = Path('/tmp/mtp-target-before-python')
after = root / 'python'
assert before.is_dir()
patch, changed = [], []
for relative in sorted({p.relative_to(before) for p in before.rglob('*.py')} |
                       {p.relative_to(after) for p in after.rglob('*.py')}):
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
Path('/tmp/mtp-target-runtime.patch').write_text(''.join(patch))
tracked = changed + ['tests/moe/test_offload.py', 'tests/moe/test_fused_copy.py',
                     'tests/layers/test_linear_mtp.py', 'tests/scheduler/test_cache_rebuild.py',
                     'tests/scheduler/test_mtp.py', 'benchmarks/bench_mtp.py',
                     'benchmarks/bench_mtp_chat.py', 'benchmarks/bench_mtp_linear.py',
                     'benchmarks/bench_mtp_copy.py', 'benchmarks/mtp_diagnostics.py']
report = dict(branch='feat/mtp-support', base_head='3381015664c3acc5003b04fcad9a29742429792c',
              patch_reference='Pre-additional-optimization runtime snapshot; includes prior uncommitted work',
              versions={name: importlib.metadata.version(name)
                        for name in ('torch', 'triton', 'transformers')},
              python=platform.python_version(), runtime_changed=changed,
              sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in tracked})
Path('/tmp/mtp-target-source.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(changed=changed, versions=report['versions']), indent=2))
