import json
import math

# Set the input JSONL file path here
INPUT_FILE = "validation_20250818_072504.jsonl"

latencies_by_step = {}

# Read JSONL and group latencies by step
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    for raw_line in file:
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "step" not in obj or "latency_sec" not in obj:
            continue

        try:
            step_number = int(obj["step"])  # ensure step is an int
        except (TypeError, ValueError):
            continue

        try:
            latency_seconds = float(obj["latency_sec"])  # ensure latency is a float
        except (TypeError, ValueError):
            continue

        latencies_by_step.setdefault(step_number, []).append(latency_seconds)

# If no valid data, print NaN for each percentile and exit
if not latencies_by_step:
    print("NaN")
    print("NaN")
    print("NaN")
    print("NaN")
    raise SystemExit(0)

# Order steps and keep the last 10 steps
ordered_steps = sorted(latencies_by_step.keys())
last_ten_steps = ordered_steps[-10:]

# Collect all latencies from the last 10 steps
combined_latencies = []
for step in last_ten_steps:
    combined_latencies.extend(latencies_by_step[step])

if not combined_latencies:
    print("NaN")
    print("NaN")
    print("NaN")
    print("NaN")
    raise SystemExit(0)

# Compute percentiles using the nearest-rank method
combined_latencies.sort()
n = len(combined_latencies)
p_values = [0.50, 0.90, 0.95, 0.99]
results = []
for p in p_values:
    rank = math.ceil(p * n)
    index = max(0, min(n - 1, rank - 1))  # convert 1-based rank to 0-based index
    results.append(combined_latencies[index])

# Output values in order: P50, P90, P95, P99 (one per line)
for value in results:
    print(value)


