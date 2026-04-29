"""Fast sweep of all 400 tasks with per-task subprocess timeout."""
import sys, os, time, subprocess, json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = str(Path(__file__).parent.parent)
TASK_DIR = Path('/Users/bharath/neurogolf-2026')
TIMEOUT = 5.0

WORKER = """
import sys
sys.path.insert(0, {src!r})
sys.path.insert(0, {scripts!r})
from neurogolf.task_io import load_task_json
from search_solvers import find_master_synthesis
import json
task = load_task_json(sys.argv[1])
model = find_master_synthesis(task, max_shift=2)
print(json.dumps(type(model).__name__ if model else None))
"""

tasks = sorted(TASK_DIR.glob('task*.json'))
solved, failed, timed_out, errors = [], [], [], []

src_dir = os.path.join(PROJECT_ROOT, 'src')
scripts_dir = os.path.join(PROJECT_ROOT, 'scripts')
worker_code = WORKER.format(src=src_dir, scripts=scripts_dir)

for i, t in enumerate(tasks):
    try:
        r = subprocess.run(
            [sys.executable, '-c', worker_code, str(t)],
            capture_output=True, text=True, timeout=TIMEOUT,
            cwd=PROJECT_ROOT
        )
        if r.returncode == 0 and r.stdout.strip():
            result = json.loads(r.stdout.strip())
            if result:
                solved.append((t.stem, result))
            else:
                failed.append(t.stem)
        else:
            errors.append(t.stem)
    except subprocess.TimeoutExpired:
        timed_out.append(t.stem)
    except Exception as e:
        errors.append(f'{t.stem}:{e}')

    if (i + 1) % 20 == 0:
        print(f'{i+1}/400: solved={len(solved)} timeout={len(timed_out)} fail={len(failed)} err={len(errors)}', flush=True)

print(f'\nFINAL: solved={len(solved)} timeout={len(timed_out)} failed={len(failed)} err={len(errors)}')
cnt = Counter(s[1] for s in solved)
print('Solvers:', dict(cnt.most_common(10)))
print('Solved:', sorted(s[0] for s in solved))
print('TimedOut:', timed_out[:30])
