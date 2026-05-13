#!/usr/bin/env python3
import sys, yaml, pathlib
base, out, *pairs = sys.argv[1:]
P = yaml.safe_load(open(base))

def parse_value(val):
    low = val.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return float(val) if ('.' in val or 'e' in val.lower()) else int(val)
    except Exception:
        return val

def set_by_dotted(d, path, val):
    parts = path.split('.')
    cur = d
    for k in parts[:-1]:
        cur = cur.setdefault(k, {})
    cur[parts[-1]] = parse_value(val)

for kv in pairs:
    k, v = kv.split('=',1)
    set_by_dotted(P, k, v)

pathlib.Path(out).write_text(yaml.safe_dump(P, sort_keys=False))
print(f"Wrote {out}")
