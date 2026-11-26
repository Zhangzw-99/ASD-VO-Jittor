#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

if [ -d "/usr/lib/wsl/lib" ]; then
  case ":$PATH:" in
    *":/usr/lib/wsl/lib:"*) ;;
    *) export PATH="/usr/lib/wsl/lib:$PATH" ;;
  esac
  case ":${LD_LIBRARY_PATH:-}:" in
    *":/usr/lib/wsl/lib:"*) ;;
    *) export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
fi

if [ "$#" -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "Usage: bash scripts/run.sh --videos-root /path/to/input --output-root /path/to/output [extra args...]" >&2
  if [ "$#" -eq 0 ]; then
    exit 1
  fi
  exec python "${repo_root}/climb/run_climb_submission.py" --help
fi

default_network="${repo_root}/ckpt/dpvo.pth"
if [ ! -f "${default_network}" ] && [ -f "${repo_root}/ckpt/dpvo.pkl" ]; then
  default_network="${repo_root}/ckpt/dpvo.pkl"
fi

exec python "${repo_root}/climb/run_climb_submission.py" \
  --network "${default_network}" \
  --config "${repo_root}/configs/asd_vo.yaml" \
  --runs 1 2 3 4 5 \
  --base-seed 12345 \
  --scale 0.5 \
  --undistort fisheye \
  "$@"
