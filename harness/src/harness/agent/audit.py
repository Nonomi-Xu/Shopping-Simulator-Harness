"""Append-only full-chain records, flushed before and after each external call.

The callback owns one task directory. Headers and credentials are never recorded.
Large histories are stored per task, not retained for an entire benchmark.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def redact(value):
    if isinstance(value, dict):
        return {k: ('***' if any(s in k.lower() for s in ('api_key', 'authorization', 'password', 'secret'))
                    else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(',', ':'))
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class RunAudit:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.started = time.monotonic()
        self.sequence = 0
        self.active_request = None
        (self.directory/'events.jsonl').touch()

    def append(self, filename, payload):
        with (self.directory/filename).open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n')
            handle.flush()

    def __call__(self, event):
        self.append('events.jsonl', event.model_dump(mode='json'))
        self.append('timeline.jsonl', {'event_id': event.event_id, 'kind': event.kind,
                    'utc': datetime.now(timezone.utc).isoformat(),
                    'elapsed_seconds': time.monotonic() - self.started})

    def transport(self, kind, payload):
        if kind == 'request':
            self.sequence += 1
            self.active_request = self.sequence
        self.append('model_io.jsonl', {
            'kind': kind, 'request_id': self.active_request,
            'utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - self.started,
            'payload': redact(payload),
        })

    def finish(self, result):
        from .artifacts import write_run_artifacts
        # Events include every actual context, parsed response, action and observation.
        # Persist the session separately: private evaluation is never fed to the actor.
        write_json(self.directory/'session.json', result.session.model_dump(mode='json'))
        write_json(self.directory/'result.json', result.model_dump(mode='json', exclude={
            'events', 'messages', 'turns', 'session', 'memory', 'plan', 'profile', 'user_profile', 'task_profile',
        }))
        write_run_artifacts(result, self.directory, streamed=True)
