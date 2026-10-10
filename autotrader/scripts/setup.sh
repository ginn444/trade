#!/usr/bin/env bash
# Reproducible local setup: pinned dependencies in a virtualenv, then the test suite.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.lock.txt
pip install --no-deps -e .
[ -f .env ] || { cp .env.example .env; chmod 600 .env; echo "created .env — fill in your keys"; }
pytest
