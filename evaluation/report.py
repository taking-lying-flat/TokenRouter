import argparse
import csv
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--result", type=str)
args = parser.parse_args()
INPUT_CSV = args.result

rows = []
with open(INPUT_CSV, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r.get("input_id") == "input_id":
            continue
        rows.append(r)

rows = list(enumerate(rows))
rows.sort(key=lambda x: (int(x[1]["input_id"]), x[0]))
rows = [r for _, r in rows]

worker_time = defaultdict(float)
worker_tokens = defaultdict(float)

input_time = defaultdict(float)
input_tokens = defaultdict(float)
input_model_tokens = defaultdict(lambda: defaultdict(float))
global_model_tokens = defaultdict(float)


def ffloat(x):
    try:
        return float(x)
    except:
        return 0.0


def fint(x):
    try:
        return int(x)
    except:
        return 0


model_names = []
if rows:
    model_names = [
        field[:-4]
        for field in rows[0]
        if field.endswith("_acc")
    ]


for row in rows:

    wid = fint(row["worker_id"])
    iid = fint(row["input_id"])

    latency = ffloat(row["latency_ms"]) / 1000.0

    model_tokens = {
        model_name: fint(row.get(f"{model_name}_acc", 0))
        for model_name in model_names
    }
    total_tokens = sum(model_tokens.values())

    worker_time[wid] += latency
    worker_tokens[wid] += total_tokens

    input_time[iid] += latency
    input_tokens[iid] += total_tokens
    for model_name, token_count in model_tokens.items():
        input_model_tokens[iid][model_name] += token_count
        global_model_tokens[model_name] += token_count


# =========================
# input-level metrics
# =========================
input_speed = {}
input_model_ratios = {}

for iid in input_tokens:
    if input_time[iid] > 0:
        input_speed[iid] = input_tokens[iid] / input_time[iid]
    else:
        input_speed[iid] = 0.0

    denom = sum(input_model_tokens[iid].values())
    input_model_ratios[iid] = {
        model_name: (
            input_model_tokens[iid][model_name] / denom if denom > 0 else 0.0
        )
        for model_name in model_names
    }


if not rows:
    print("No rows found in input CSV, nothing to report.")
    raise SystemExit(0)

base_fields = list(rows[0].keys())

ratio_fields = [f"input_{model_name}_ratio" for model_name in model_names]
if "input_speed" not in base_fields:
    idx = base_fields.index("input_id") + 1
    fieldnames = (
        base_fields[:idx] + ["input_speed", *ratio_fields] + base_fields[idx:]
    )
else:
    fieldnames = base_fields


with open(INPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for r in rows:
        iid = fint(r["input_id"])

        r["input_speed"] = f"{input_speed.get(iid, 0.0):.6f}"
        for model_name in model_names:
            r[f"input_{model_name}_ratio"] = (
                f"{input_model_ratios.get(iid, {}).get(model_name, 0.0):.6f}"
            )

        writer.writerow(r)


print("\n================= PER WORKER =================")

system_throughput = 0.0

for wid in sorted(worker_time.keys()):
    if worker_time[wid] > 0:
        speed = worker_tokens[wid] / worker_time[wid]
    else:
        speed = 0.0

    system_throughput += speed
    print(f"Worker {wid}: {speed:.2f} tok/s")

print("\n================= SYSTEM =================")

print(f"Total throughput: {system_throughput:.2f} tok/s")

global_total = sum(global_model_tokens.values())
for model_name in model_names:
    ratio = (
        global_model_tokens[model_name] / global_total
        if global_total > 0
        else 0.0
    )
    print(f"Global {model_name}_ratio: {ratio:.6f}")

print("==========================================\n")
