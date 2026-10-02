"""Stop only the registered record-push queue and its children at the deadline."""
import json
import os
from pathlib import Path
import signal
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent / 'runs/record-push'


def processes():
    rows = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / 'status').read_text().splitlines()
            parent = int(next(line.split()[1] for line in status if line.startswith('PPid:')))
            command = (entry / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
            rows[int(entry.name)] = (parent, command)
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            pass
    return rows


def stop_queue(pid, rows):
    if pid not in rows:
        return []
    assert 'bash runs/record-push/resume' in rows[pid][1], rows[pid][1]
    descendants = []
    frontier = [pid]
    while frontier:
        parent = frontier.pop()
        children = [child for child, (ppid, _) in rows.items() if ppid == parent]
        descendants.extend(children)
        frontier.extend(children)
    # Stop the shell first so a completed child cannot start the next experiment.
    targets = [pid] + list(reversed(descendants))
    stopped = []
    for target in targets:
        try:
            os.kill(target, signal.SIGTERM)
            stopped.append({'pid': target, 'command': rows[target][1]})
        except ProcessLookupError:
            pass
    return stopped


def run():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    deadline = datetime.fromisoformat(manifest['deadline_utc'].replace('Z', '+00:00')).timestamp()
    print('Watching record-push deadline', manifest['deadline_utc'], flush=True)
    while time.time() < deadline:
        time.sleep(min(30, deadline - time.time()))
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    stopped = stop_queue(manifest['queue_pid'], processes())
    result = {'deadline_utc': manifest['deadline_utc'],
              'stopped_at_utc': datetime.now(timezone.utc).isoformat(), 'stopped_processes': stopped,
              'policy': 'Stop registered experiment queue only; preserve completed and partial artifacts'}
    (ROOT / 'deadline-stop.json').write_text(json.dumps(result, indent=2))
    print('DEADLINE STOP', json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
