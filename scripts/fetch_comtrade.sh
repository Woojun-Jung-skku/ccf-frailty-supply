#!/bin/bash
# Fetch Korea (410) monthly imports, all partners, for selected HS6 codes from the UN Comtrade public preview API.
ROOT=${PAPER_S_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=$ROOT/data_raw/comtrade
mkdir -p "$OUT"
CODES="281111,370790,392099,811292,282560,381800,280461,281290,282619,280429"
for y in $(seq 2014 2026); do
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    p="${y}${m}"
    f="$OUT/k410_${p}.json"
    if [ -s "$f" ] && grep -q '"data"' "$f"; then continue; fi
    for attempt in 1 2 3 4; do
      curl -sS --max-time 60 "https://comtradeapi.un.org/public/v1/preview/C/M/HS?reporterCode=410&period=${p}&cmdCode=${CODES}&flowCode=M" -o "$f.tmp"
      if grep -q '"data"' "$f.tmp" 2>/dev/null; then mv "$f.tmp" "$f"; break; fi
      sleep $((attempt*5))
    done
    echo "$p $(python3 -I -c "import json;d=json.load(open('$f'));print(d.get('count'),d.get('error'))" 2>/dev/null)"
    sleep 1.2
  done
done
echo DONE
