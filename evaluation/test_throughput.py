import argparse
import csv
import fcntl
import json
import os
import time

import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument("--input_id", type=int, required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--input_file",
        type=str,
        default="evaluation/datasets/swesmith_tool_qwen3_8k_messages.json",
    )
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument(
        "--model_names",
        type=str,
        default=os.getenv("TOKENROUTER_MODEL_NAMES", ""),
        help=(
            "Comma-separated model names. If omitted, infer them from the first "
            "TokenRouter usage_details response."
        ),
    )
    return parser.parse_args()


def build_round_messages(problem):
    if isinstance(problem, str):
        return [[{"role": "user", "content": problem}]]

    if not isinstance(problem, list):
        raise TypeError("`problem` must be either a prompt string or an OpenAI message list")

    rounds = []
    for idx, message in enumerate(problem):
        if message.get("role") == "assistant" and idx > 0:
            rounds.append(problem[:idx])

    if not rounds:
        raise ValueError("message-list problem has no assistant turns to replay")

    return rounds


def parse_model_names(value):
    model_names = [name.strip() for name in value.split(",") if name.strip()]
    if len(model_names) != len(set(model_names)):
        raise ValueError("--model_names must not contain duplicate names")
    return model_names


def response_usage_details(result):
    usage_details = result.get("usage_details")
    if usage_details is None:
        return []
    if not isinstance(usage_details, list) or any(
        not isinstance(usage, dict) for usage in usage_details
    ):
        raise ValueError("response usage_details must be a list of mappings")
    return usage_details


def resolve_model_names(result, configured_model_names):
    usage_details = response_usage_details(result)
    response_model_names = []
    for usage in usage_details:
        for model_name in usage:
            if model_name not in response_model_names:
                response_model_names.append(model_name)

    if response_model_names:
        if configured_model_names and set(configured_model_names) != set(
            response_model_names
        ):
            raise ValueError(
                "configured model names do not match response usage_details: "
                f"configured={configured_model_names}, "
                f"response={response_model_names}"
            )
        return configured_model_names or response_model_names

    if configured_model_names:
        return configured_model_names

    # Preserve the legacy standard-OpenAI endpoint behavior. Such a response
    # has no per-model information, so its completion is treated as reference.
    if result.get("usage") is not None:
        return ["quick", "reference"]

    raise ValueError("response contains neither usage_details nor usage")


def extract_usage(result, model_names):
    usage_details = response_usage_details(result)
    extracted = {
        f"{model_name}_{suffix}": 0
        for model_name in model_names
        for suffix in ("acc", "gen")
    }

    if not usage_details:
        usage = result.get("usage")
        if usage is None:
            raise ValueError("response contains neither usage_details nor usage")
        if model_names != ["quick", "reference"] and len(model_names) != 1:
            raise ValueError(
                "standard OpenAI usage cannot be attributed across multiple models; "
                "the endpoint must return usage_details"
            )
        completion = usage.get("completion_tokens", 0)
        final_model = model_names[-1]
        extracted[f"{final_model}_acc"] = completion
        extracted[f"{final_model}_gen"] = completion
        return extracted

    for usage in usage_details:
        missing = set(model_names) - set(usage)
        extra = set(usage) - set(model_names)
        if missing or extra:
            raise ValueError(
                "model names changed between responses: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for model_name in model_names:
            model_usage = usage[model_name]
            extracted[f"{model_name}_acc"] += model_usage.get(
                "accepted_tokens", 0
            )
            extracted[f"{model_name}_gen"] += model_usage.get(
                "generated_tokens", 0
            )
    return extracted


def response_output_file(output_file):
    output_dir = os.path.dirname(os.path.abspath(output_file))
    output_stem = os.path.splitext(os.path.basename(output_file))[0]
    return os.path.join(output_dir, f"{output_stem}_responses.jsonl")


def main():
    args = parse_args()
    configured_model_names = parse_model_names(args.model_names)
    model_names = None

    with open(args.input_file, "r") as f:
        dataset = json.load(f)

    problem = dataset[args.input_id]["problem"]
    round_messages = build_round_messages(problem)

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"

    first_start_ms = None
    last_end_ms = None
    total_latency_ms = 0
    response_records = []
    totals = None

    for round_id, messages in enumerate(round_messages):
        payload = {
            "model": "default",
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": args.max_tokens,
            "stream": False,
        }

        start = time.time()
        start_ms = int(start * 1000)
        if first_start_ms is None:
            first_start_ms = start_ms

        resp = requests.post(
            url, json=payload, timeout=int(os.getenv("REQUEST_TIMEOUT", "3600"))
        )

        end = time.time()
        end_ms = int(end * 1000)
        last_end_ms = end_ms
        total_latency_ms += end_ms - start_ms

        resp.raise_for_status()
        result = resp.json()

        response_records.append(
            {
                "worker_id": args.worker_id,
                "input_id": args.input_id,
                "round_id": round_id,
                "content": result["choices"][0]["message"]["content"],
            }
        )

        round_model_names = resolve_model_names(
            result, model_names or configured_model_names
        )
        if model_names is None:
            model_names = round_model_names
            totals = {
                f"{model_name}_{suffix}": 0
                for model_name in model_names
                for suffix in ("acc", "gen")
            }

        usage = extract_usage(result, model_names)
        for key, value in usage.items():
            totals[key] += value

    if model_names is None or totals is None:
        raise ValueError("no evaluation rounds were generated")

    row = [
        args.worker_id,
        args.input_id,
        first_start_ms,
        last_end_ms,
        total_latency_ms,
        *[
            totals[f"{model_name}_{suffix}"]
            for model_name in model_names
            for suffix in ("acc", "gen")
        ],
    ]

    if args.record == 1:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

        with open(args.output_file, "a+", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            expected_header = [
                "worker_id",
                "input_id",
                "start_ms",
                "end_ms",
                "latency_ms",
                *[
                    f"{model_name}_{suffix}"
                    for model_name in model_names
                    for suffix in ("acc", "gen")
                ],
            ]

            f.seek(0)
            existing_header = next(csv.reader(f), None)
            writer = csv.writer(f)

            if existing_header is None:
                writer.writerow(expected_header)
            elif existing_header != expected_header:
                raise ValueError(
                    "result CSV header does not match response model names: "
                    f"header={existing_header}, expected={expected_header}"
                )

            f.seek(0, os.SEEK_END)
            writer.writerow(row)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

        with open(
            response_output_file(args.output_file), "a+", encoding="utf-8"
        ) as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            for response_record in response_records:
                f.write(json.dumps(response_record, ensure_ascii=False) + "\n")
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
