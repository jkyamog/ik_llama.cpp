# CPU NUMA Tensor Parallel Design Hypothesis

This note is Task 1 for the CPU NUMA tensor-parallel spike. It is a hypothesis to validate against the current code paths before implementation.

## Goal

Expose two CPU NUMA sockets as tensor-parallel compute targets so CPU-resident model tensors can be split across sockets and decoded with better memory locality and bandwidth.

The first target is Linux with exactly two usable NUMA nodes. The feature must be disabled by default and gated behind an experimental `--cpu-tp 2` option.

## Representation Hypothesis

Represent CPU NUMA nodes as both:

- scheduler-visible CPU backend instances, so graph execution can target CPU node 0 or CPU node 1 independently;
- synthetic llama model devices, so model loading and split-mode graph logic can assign tensor shards to those backends.

Backend registry entries alone are not enough. The llama model/device layer must also see CPU node devices, otherwise split tensor placement continues to treat CPU as a fallback backend instead of a split target.

## CPU Node 0 vs Fallback CPU

Keep the existing CPU backend as the compatibility fallback used when `--cpu-tp` is disabled.

With `--cpu-tp 2`, create separate CPU-node backends:

- `CPU-NUMA0`: compute pinned to NUMA node 0, memory placed or first-touched on NUMA node 0.
- `CPU-NUMA1`: compute pinned to NUMA node 1, memory placed or first-touched on NUMA node 1.

Do not silently replace the normal `CPU` backend globally. The fallback CPU path should remain available for unsupported ops, normal no-flag runs, and error recovery.

## Buffer Type Hypothesis

Each CPU node needs a distinct buffer type or buffer context carrying a NUMA node id. A shared host CPU buffer type is too ambiguous because the scheduler and allocation logs cannot tell which node owns the memory.

Minimum viable behavior:

- CPU node buffer allocation uses node-local allocation if available.
- If direct node-local allocation is not available, allocation performs explicit first-touch from threads pinned to the intended node.
- Buffer names include the node id for logging and debugging.

The implementation should first try to reuse existing `/sys/devices/system/node` NUMA detection patterns. Add a libnuma dependency only if the existing code cannot provide reliable node-local allocation or first-touch.

## Threading and Affinity Hypothesis

Each CPU-node backend should carry:

- NUMA node id.
- CPU set for that node.
- Thread count for that node.

For v1, use a simple split:

- total requested decode threads divided evenly across the two CPU-node backends;
- if an odd count is provided, node 0 receives the extra thread;
- batch/prefill threads follow the same rule unless Task 2 finds separate scheduler plumbing for batch vs decode.

The implementation must not globally pin the process in a way that prevents either node from running. Existing `--numa` modes should be treated as separate behavior:

- `--cpu-tp 0`: existing `--numa` behavior is unchanged.
- `--cpu-tp 2`: CPU TP owns backend-local affinity. Combining `--cpu-tp 2` with `--numa distribute`, `isolate`, or `numactl` should initially be rejected or logged as unsupported until validated.

## Split-Mode Path Hypothesis

The intended reuse path is split-mode graph, not layer-only splitting.

Expected flow:

1. `--cpu-tp 2` exposes two synthetic CPU devices to llama model initialization.
2. Model splits are computed as `{0.5, 0.5}` for CPU-only v1.
3. Split tensors are created by the existing split tensor machinery.
4. Each tensor shard is assigned to the matching CPU-node buffer/backend.
5. The backend scheduler executes each shard on its node-local backend.

Mixed GPU plus CPU-node tensor parallelism is out of scope for v1 unless Task 2 shows it falls out naturally without device-index ambiguity. If device indexing gets messy, v1 should reject mixed CPU TP and GPU TP.

## Reduction Hypothesis

Partial matmul outputs must be combined into the same logical result as the baseline unsplit CPU path.

The optimistic hypothesis is that existing split-mode graph reduction machinery can be reused once CPU-node tensors behave like real split devices. If that is false, the fallback design is a narrow CPU split-output reduce op used only for tensors that split matmul output.

Do not implement broad GGML op rewrites in this spike. If correct CPU reduction requires broad op coverage, stop and report the blocker.

## Compatibility Rules

- Default no-flag behavior must be unchanged.
- `--cpu-tp 0` is identical to current behavior.
- `--cpu-tp 1` is accepted as disabled-equivalent with an explicit log.
- `--cpu-tp 2` only runs when two usable NUMA nodes are detected.
- Unsupported systems fail early with a clear message.
- gpu-cpu-hybrid serving configs must not change until correctness and speedup are proven.

## Open Questions For Task 2

- Can existing split-mode graph reductions operate on CPU backends once CPU nodes are exposed as devices?
- Does the current scheduler allow multiple CPU backends with distinct thread counts and affinity?
- Are split tensors currently keyed by device index in a way that assumes GPU-like devices?
- Is a distinct CPU-node buffer type enough, or is a true CPU split buffer type needed?
- Where should `--cpu-tp` live in public model params so server and CLI share it cleanly?

## Task 2 Validation

Verdict: the hypothesis still looks viable, but only with real plumbing changes in the llama device model, CPU buffer types, and CPU backend instantiation. It is not just a CLI flag or NUMA-affinity wrapper.

Validated code paths:

- CPU backend registration is currently singleton-shaped. `ggml/src/ggml-backend.cpp` registers `"CPU"` in `ggml_backend_register()` setup, exposes the singleton buffer type through `ggml_backend_cpu_buffer_type()`, and creates CPU backends through `ggml_backend_cpu_init()`. CPU TP needs distinct CPU-node buffer types or buffer contexts such as `CPU-NUMA0` and `CPU-NUMA1`; otherwise `ggml_backend_sched_backend_from_buffer()` cannot distinguish node-local ownership.
- NUMA detection exists but is global. `ggml/src/ggml.c` implements `ggml_numa_init()`, `ggml_is_numa()`, `set_numa_thread_affinity()`, and `clear_numa_thread_affinity()`. This can supply topology and CPU sets, but CPU TP should not reuse the global `--numa distribute/isolate/numactl` behavior directly because per-node backends need independent affinity state.
- Device counting excludes CPU. `src/llama.cpp` has `llama_get_device_count()` for CUDA/SYCL/Vulkan/CANN/RPC-like devices; CPU is treated as fallback. CPU TP needs an explicit path that contributes synthetic CPU-node devices only when `--cpu-tp 2` is active.
- Buffer selection is GPU-oriented for split mode. `src/llama.cpp` has `llama_default_buffer_type_cpu()`, `llama_default_buffer_type_offload()`, and `llama_default_buffer_type_split()`. The split buffer path currently handles CUDA/SYCL split buffer types, then falls back through offload buffer selection. CPU TP needs a CPU-node buffer selection path and likely a CPU-node split buffer type or equivalent.
- The model-side containers are generic enough. `src/llama-model.h` stores `split_mode`, `devices`, and split-mode tensor accounting without being inherently GPU-only. `src/llama.cpp` populates `model.devices`, derives `model.splits`, and assigns `model.buft_layer`; those paths can represent CPU-node devices if the device IDs and buffer types are made unambiguous.
- Split tensor creation is mostly reusable. `src/llama-load-tensors.cpp` uses `ctx_split`, `model.splits`, and split tensor helpers for `LLAMA_SPLIT_MODE_GRAPH` and `LLAMA_SPLIT_MODE_ATTN`. The creation path is generic, but actual placement still depends on the buffer type and scheduler mapping being correct.
- Scheduler placement is buffer-type driven. `ggml/src/ggml-backend.cpp` uses `ggml_backend_sched_backend_from_buffer()` and backend support checks to map tensors to backends. Multiple CPU-node backends are plausible if each advertises a distinct supported buffer type. Watch for CPU-last assumptions such as the scheduler asserting the final backend is CPU.
- CLI/API plumbing belongs in the shared common/model params path. `common/common.cpp` already parses `--numa`, split mode, `--tensor-split`, `--device`, and writes `params.devices` into `llama_model_params` via `common_model_params_to_llama()`. `include/llama.h` exposes `llama_model_params::devices`, `split_mode`, and `tensor_split`; `--cpu-tp` should be added to the shared params so CLI and server initialization agree.

Validation checks:

- Device model: viable if CPU-node device IDs are added only behind `--cpu-tp 2` and v1 rejects mixed GPU/RPC plus CPU TP to avoid ambiguous numbering.
- Backend/scheduler: viable if two CPU backend instances can advertise separate buffer types and keep separate affinity/thread state. This needs implementation, not just reuse of the singleton CPU backend.
- Buffer placement: viable with either node-local allocation or first-touch allocation from node-pinned threads. Existing NUMA discovery helps, but no current public CPU buffer type carries a node id.
- Split tensor path: viable in principle because split tensors are generated from `model.splits` and `ctx_split`, but CPU-node split buffer selection must be added before it can target CPU nodes.
- Reduction path: likely viable through existing split-mode graph mechanics if CPU-node tensors behave like real split devices. If Task 6 finds CUDA/SYCL-only assumptions in reduction, stop rather than broadening GGML op rewrites.
- CLI/API path: add `cpu_tp` to the shared common params and propagate into model initialization. Reject `--cpu-tp 2` with existing `--numa` modes for v1 until interaction is validated.

Task 3 recommendation: proceed with experimental CLI plumbing only. Add `--cpu-tp N` with `0` default, `1` disabled-equivalent logging, `2` request mode, and clear rejection for any other value. Task 3 should only store and validate the request; it should not create CPU-node backends yet.

## Task 5 Revision Status

Task 5 established distinct CPU-node identities but did not complete true NUMA-local execution.

Implemented:

- Linux builds can create distinct `CPU-NUMA0` and `CPU-NUMA1` CPU buffer types.
- CPU TP model loading uses those buffer types for synthetic CPU-node devices.
- CPU TP context initialization creates named CPU backend instances for each synthetic node.
- Invalid NUMA buffer type requests are bounded to avoid out-of-range access.

Not implemented:

- Per-node worker-thread affinity.
- Guaranteed NUMA-local allocation or first-touch placement.

Blocker:

`ggml_backend_cpu_graph_compute()` ultimately calls `ggml_graph_compute()` in `ggml/src/ggml.c`, where worker threads are created and `set_numa_thread_affinity()` applies the global `--numa` strategy. That path does not receive backend-local NUMA node state, so calling an affinity helper at the backend boundary would only affect the caller thread, not the worker threads. Completing Task 5 requires carrying a per-backend NUMA node through `ggml_cplan` or an equivalent compute API into `ggml_graph_compute_thread()`.

Until that is done, `CPU-NUMA*` buffer/backend names are placement plumbing and auditability only; they should not be treated as proof of node-local memory or compute.
