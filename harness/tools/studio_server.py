"""Run the local Studio Agent Server; optionally detach with logs under outputs/studio."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from src.harness.env import load_dotenv
from src.harness.paths import HARNESS_DIR, OUTPUTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--port', type=int, default=2024)
    parser.add_argument('--env-file', type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    local_env = args.env_file or HARNESS_DIR / '.env'
    if args.env_file or local_env.exists():
        load_dotenv(local_env)
    executable = Path(sys.executable).parent / 'langgraph'
    if not executable.is_file():
        parser.error("install .[studio] in this Python environment first")
    command = [str(executable), 'dev', '--config', str(HARNESS_DIR / 'langgraph.json'),
               '--host', '127.0.0.1', '--port', str(args.port), '--no-browser', '--no-reload']
    # Refuse a duplicate service without touching the existing listener.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', args.port))
        except OSError as exc:
            parser.error(f'cannot listen on 127.0.0.1:{args.port}: {exc}')
    if not args.background:
        os.chdir(HARNESS_DIR)
        os.execv(str(executable), command)
    directory = OUTPUTS_DIR / 'studio'
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f'server-{args.port}.log'
    with log.open('ab') as handle:
        process = subprocess.Popen(command, cwd=HARNESS_DIR, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    record = {'pid': process.pid, 'port': args.port, 'log': str(log),
              'studio_url': f'https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:{args.port}'}
    (directory / f'server-{args.port}.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
