#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [-A <allocation_name>]" >&2
  echo "  -A <allocation_name>   Override allocation (default: ~/.idevrc idev_project)" >&2
}

ALLOC=""

while getopts ":A:h" opt; do
  case "$opt" in
    A)
      ALLOC="$OPTARG"
      ;;
    h)
      usage
      exit 0
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      exit 1
      ;;
    \?)
      echo "Unknown option: -$OPTARG" >&2
      usage
      exit 1
      ;;
  esac
done

shift "$((OPTIND - 1))"

if [[ $# -ne 0 ]]; then
  usage
  exit 1
fi

if [[ -z "$ALLOC" ]]; then
  if [[ ! -f "$HOME/.idevrc" ]]; then
    echo "No -A provided and ~/.idevrc not found." >&2
    exit 1
  fi

  ALLOC="$(awk '$1 == "idev_project" {print $2; exit}' "$HOME/.idevrc")"
  if [[ -z "$ALLOC" ]]; then
    echo "No -A provided and no 'idev_project' entry found in ~/.idevrc." >&2
    exit 1
  fi
fi

# Slurm account names on Stampede3 are lowercase in assoc_mgr output.
ALLOC="${ALLOC,,}"

scontrol -o show assoc_mgr account="$ALLOC" \
| awk -v a="$ALLOC" '
match($0,/Account=([^ ]+)/,am) && am[1] == a &&
match($0,/UserName=([^ (]+)/,um) &&
match($0,/UsageRaw\/Norm\/Efctv=([^\/]+)/,rm) {
  print um[1] "|" rm[1]
}
' \
| sort -t"|" -k2,2nr
