# TokenRouter: Efficient Serving System for Token-Level LLM Routing

TokenRouter is an efficient and user-friendly serving system for token-level routed LLM inference. TokenRouter shifts from the conventional request-centric paradigm to a model-centric one. Instead of treating each request as a synchronized decoding stream, TokenRouter launches a subserver for each candidate LLM and dispatches requests asynchronously according to routing decisions. Each subserver employs a delayed-batching scheduler, whose hyperparameters are derived mathematically from a throughput model of the system. Across diverse routing algorithms, workloads, and model pairs, TokenRouter achieves 2.01-64.15× higher decoding throughput than existing systems.

## Supported Token-level Routing Algorithms

| Method | Scheduler Classes |
| --- | --- |
| R2R | `r2r_slm` + `r2r_llm` |
| CITER | `citer_slm` + `citer_llm` |
| Co-LLM | `collm_base` + `collm_assist` |
| R-Stitch | `rstitch_slm` + `rstitch_llm` |
| GlimpRouter | `glimprouter_slm` + `glimprouter_llm` |
| ME | `random_routing` on `small` + `medium` + `large` |
| Query-level routing | `query_level_routing` |
| Random baseline | `random_routing` |

## Installation

```bash
conda create -n tokenrouter python==3.10
conda activate tokenrouter
pip install -e .
```

## Quick Start

TokenRouting is configured through a single YAML file. Example configs are provided under [config/](config/), and runnable usage examples are provided under [examples/](examples/).

### Server Mode

Launch a TokenRouter server with a routing config:

```bash
python -m tokenrouter.launch_server --config_path config/r2r/Qwen3-0.6B_Qwen3-32B.yaml
```

To place one model on each node, keep both model definitions in the shared
config, add `system.peer_endpoints`, and select the model launched by each
process:

```yaml
system:
  entry_model: quick
  peer_endpoints:
    quick:
      bind: "tcp://0.0.0.0:30000"
      advertise: "tcp://{$REFERENCE_IP}:30000"
    reference:
      bind: "tcp://0.0.0.0:30000"
      advertise: "tcp://{$QUICK_IP}:30000"
```

```bash
# On quick node
python -m tokenrouter.launch_server --config_path config.yaml --local_model quick

# On reference node
python -m tokenrouter.launch_server --config_path config.yaml --local_model reference
```

Once the server is running, send a chat completion request with either the OpenAI client example or a plain `requests` example:

```bash
python examples/openai_chat_completion.py
python examples/requests_chat_completion.py
```

### Engine Mode

Use engine mode when you want to run TokenRouter directly from Python without starting a separate HTTP server. The example below creates a `TokenRoutingEngine` from the same YAML config and calls `engine.generate()`:

```bash
python examples/engine_generate.py
```

## Reproducing Our Results

Select one config and one workload before launching the server.

| Method | Config path |
| --- | --- |
| R2R | [config/r2r/Qwen3-0.6B_Qwen3-32B.yaml](config/r2r/Qwen3-0.6B_Qwen3-32B.yaml) |
| CITER | [config/citer/Qwen3-0.6B_Qwen3-32B.yaml](config/citer/Qwen3-0.6B_Qwen3-32B.yaml) |
| Co-LLM | [config/collm/Qwen3-0.6B_Qwen3-32B.yaml](config/collm/Qwen3-0.6B_Qwen3-32B.yaml) |
| R-Stitch | [config/rstitch/Qwen3-0.6B_Qwen3-32B.yaml](config/rstitch/Qwen3-0.6B_Qwen3-32B.yaml) |
| ME | [config/me/Qwen3-0.6B_Qwen3-8B_Qwen3-32B.yaml](config/me/Qwen3-0.6B_Qwen3-8B_Qwen3-32B.yaml) |

| Workload | Dataset path |
| --- | --- |
| AIME | [evaluation/datasets/aime.json](evaluation/datasets/aime.json) |
| AIME 8K output budget | [evaluation/datasets/aime_8k.json](evaluation/datasets/aime_8k.json) |
| SWE-smith multi-round trajectories | [evaluation/datasets/swesmith_traj_multi_round.json](evaluation/datasets/swesmith_traj_multi_round.json) |

### 1. Launch CUDA MPS

If multiple models share the same physical GPU, CUDA MPS is recommended for reproducing overlapping serving behavior and improving concurrent scheduling performance.

```bash
mkdir -p tmp/mps_${METHOD}_${PORT}

export CUDA_MPS_PIPE_DIRECTORY=$PWD/tmp/mps_${METHOD}_${PORT}/pipe
export CUDA_MPS_LOG_DIRECTORY=$PWD/tmp/mps_${METHOD}_${PORT}/log

nvidia-cuda-mps-control -d
```

### 2. Launch Server

Start the TokenRouter server with the selected routing config.

```bash
python -m tokenrouter.launch_server --config_path $CONFIG_PATH
```

### 3. Run Throughput Benchmark

Run the workload on the server and write per-request measurements to the output directory.

```bash
bash evaluation/test_throughput.sh \
  $CONCURRENCY \
  $PORT \
  $NUM_INPUTS \
  $REPEATS_PER_INPUT \
  $MAX_TOKENS \
  $OUTPUT_DIR \
  $DATASET_PATH
```

The results will be written to `${OUTPUT_DIR}/result.csv`.

Use `evaluation/report.py` to summarize an existing result CSV without rerunning the benchmark.

```bash
python evaluation/report.py --result ${OUTPUT_DIR}/result.csv
```

## Implementing A New Token-level Routing Method

TokenRouter represents each cooperating model with its own scheduler class. A new routing method usually adds one scheduler for each model. Every scheduler should subclass `BaseTokenRoutingScheduler`, defined in [tokenrouter/managers/schedulers/base_scheduler.py](tokenrouter/managers/schedulers/base_scheduler.py). Place scheduler classes in [tokenrouter/managers/schedulers/](tokenrouter/managers/schedulers/) and keep method-specific helper logic under [tokenrouter/algorithms/](tokenrouter/algorithms/). For each scheduler class, implement three key interfaces: `route()`, `send()`, and `receive()`.

### 1. Implement `route()`

`route(batch, result)` is called after each forward pass and must return a 1-D `torch.Tensor` with shape `(batch.batch_size(),)`. Each element corresponds to the request at the same index in `batch.reqs`.

Return one routing decision per request:

- Return a model name to route explicitly, for example `"small"`,
  `"medium"`, or `"large"`. Returning the current model name continues
  decoding locally.
- For existing two-model schedulers, `True` / `1` still delegates to the
  only peer and `False` / `0` still decodes locally.

### 2. Implement `send()`

`send(req, destination)` builds the `PeerReq` message sent to an addressed
scheduler. Use `PeerReq.construct_from_req(...)` to convert the local `Req`
into a peer message, passing the source scheduler name, destination scheduler
name, and the token ids that should be transferred:

```python
current_token_ids = req.origin_input_ids + req.output_ids[:-1]
req_to_send = PeerReq.construct_from_req(
    req=req,
    source=self.server_args.server_name,
    destination=destination,
    current_token_ids=current_token_ids,
    entry_model=self.server_args.entry_model_name,
    handoff_seq=req.handoff_seq,
)
```

The default `send()` implementation uses `req.origin_input_ids + req.output_ids` as `current_token_ids`, so the peer receives the prompt plus all generated tokens currently owned by this scheduler.

### 3. Implement `receive()`

`receive(msgs)` processes `PeerReq` messages from the peer scheduler. By default, `receive()` accepts all tokens sent by the peer. Override `receive()` when the method needs custom preprocessing before applying peer tokens.

### 4. Register Scheduler

Register the scheduler in [tokenrouter/managers/schedulers/__init__.py](tokenrouter/managers/schedulers/__init__.py) by adding it to `SCHEDULER_REGISTRY`. The registry key becomes the YAML config value for `scheduler_class`.

### 5. Optimize Delayed-batching Threshold

The delayed-batching threshold can be optimized using [evaluation/calculate_delayed_batching.py](evaluation/calculate_delayed_batching.py). For example, to optimize the delayed-batching threshold of R2R on Qwen3-0.6B and Qwen3-32B, run the following command:

```bash
python evaluation/calculate_delayed_batching.py \
        --M 2 --N $CONCURRENCY --latencies 6.0 27.9 \
        --P '[[0.35, 0.65], [1, 0]]' \
        --C '[[1, 0], [1, 0]]' \
        --delta 0.1
```

### 6. Add Config

Each cooperating model is configured independently inside the YAML config. TokenRouter extends `sglang.srt.server_args.ServerArgs` with additional routing-specific fields defined in [tokenrouter/server_args.py](tokenrouter/server_args.py). All the supported model-level configuration keys are listed in [config/supported_server_args.yaml](config/supported_server_args.yaml).
