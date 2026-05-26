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

## Post Task 5b Review

Task 5b added a backend-local NUMA node field to `ggml_cplan` and passes that value into worker-thread affinity. That is necessary, but review found the branch is still not meaningful CPU tensor parallelism. The current changes can make startup logs look promising while most runtime behavior remains ordinary CPU execution.

Findings:

1. CPU-NUMA tensor ownership can collapse onto the first CPU backend.

   `ggml_backend_cpu_supports_buft()` still accepts any host buffer type. Since `CPU-NUMA0`, `CPU-NUMA1`, and fallback `CPU` are all CPU backends, `ggml_backend_sched_backend_from_buffer()` can pick the first CPU backend that supports the op even when the tensor buffer is `CPU-NUMA1`. CPU-node backends need NUMA-aware buffer support: a node backend should claim its own `CPU-NUMA*` buffer type, and fallback CPU should not steal those buffers.

2. `--cpu-tp 2` does not yet force or validate CPU-NUMA tensor placement.

   The flag creates synthetic CPU-NUMA device names, but with default `n_gpu_layers == 0`, `i_gpu_start == n_layer` and repeating layers stay on ordinary CPU buffers. Startup can show `CPU-NUMA0` and `CPU-NUMA1` even when representative model tensors are not placed on CPU-NUMA buffers. The flag must either establish the intended split/offload placement or reject startup unless the required placement flags are provided.

3. CPU-NUMA compute buffers still use the generic CPU buffer type.

   Scheduler setup currently assigns generic CPU host buffers to all CPU backends. CPU-node backends need matching `CPU-NUMA*` compute buffer types, otherwise temporary allocations are not node-specific and runtime logs are not enough to prove locality.

Next tasks before split/reduction:

- Make CPU backend `supports_buft` NUMA-aware.
- Make `--cpu-tp 2` actually select or require CPU-NUMA tensor placement.
- Use CPU-NUMA buffer types for CPU-NUMA scheduler compute buffers.

Do not proceed to split tensor compute/reduction or performance tests until these are fixed and representative tensor placement is auditable.

## Task A: Split-Buffer Contract Map

Verdict: the generic split tensor model is reusable for CPU-NUMA, but the
existing reduction path is not sufficient for CPU today. CPU-NUMA needs a real
split buffer type plus a CPU implementation or CPU-specific hook for
`GGML_OP_REDUCE` add before `--cpu-tp 2` can be made correct.

Required split buffer callbacks and behavior:

- `ggml_backend_buffer_type_i::get_name`: expose a distinct parent split buffer
  type name such as `CPU-NUMA-Split`.
- `ggml_backend_buffer_type_i::alloc_buffer`: create the parent logical split
  buffer. Like CUDA, this buffer does not allocate one contiguous backing store
  for all shards; shard allocation happens per tensor during `init_tensor`.
- `ggml_backend_buffer_type_i::get_alignment`: return an alignment compatible
  with CPU tensor allocation.
- `ggml_backend_buffer_type_i::get_alloc_size`: sum the allocation sizes of all
  present shard tensors from `ggml_split_tensor_t::splits`.
- `ggml_backend_buffer_type_i::is_host`: the parent split buffer should not be
  reported as a plain host buffer, otherwise split-mode loading skips it through
  the existing host-buffer guard.
- `ggml_backend_buffer_i::get_base`: return a dummy non-null address for the
  logical parent, matching the CUDA split buffer pattern.
- `ggml_backend_buffer_i::init_tensor`: inspect `tensor->extra` as
  `ggml_split_tensor_t`, allocate each shard tensor on the matching
  `CPU-NUMA<i>` child buffer type, set `split->data`, set `split->buffer`, and
  mark shard buffers as weight buffers.
- `ggml_backend_buffer_i::set_tensor`: load an entire logical split tensor at
  offset 0, dispatching bytes into per-node shard tensors according to
  `split_dim`. It must support the same narrow forms needed by the first tensor
  family under test before it is generalized.
- `ggml_backend_buffer_i::get_tensor`: only needed for diagnostics and parity
  checks at first. It can be implemented for simple contiguous split forms and
  abort for repacked or explicit-range layouts until those are required.
- `ggml_backend_buffer_i::clear`: may be a no-op for weight buffers.

Required metadata and ownership rules:

- Split structure is generic: `ggml_split_tensor_t` in `ggml/include/ggml.h`
  carries `n_device`, `split_dim`, and the per-device `splits` array.
- llama owns the lifetime of `llama_split_tensor::tensor_splits`; the logical
  parent tensor stores `tensor->extra = &split_tensor.ggml`.
- The parent logical tensor is allocated in a split buffer type. Each child
  shard tensor must be allocated in a node-specific `CPU-NUMA<i>` child buffer
  type so scheduler buffer ownership resolves to the matching CPU-node backend.
- `CPU-NUMA<i>` backends may support only their matching node buffer type.
  Fallback `CPU` must not claim CPU-NUMA child buffers ahead of the owner.
- Default CPU buffers remain unchanged when `--cpu-tp` is disabled.

Exact creation and loading symbols:

- `ggml/include/ggml.h`: `ggml_split_tensor_t` defines split metadata.
- `ggml/src/ggml-cuda.cu`: `ggml_backend_cuda_split_buffer_*` is the reference
  contract for split parent allocation, per-shard child allocation, tensor load,
  and aggregate allocation sizing.
- `ggml/include/ggml-cuda.h` and `ggml/include/ggml-sycl.h`: public split buffer
  type factories are `ggml_backend_cuda_split_buffer_type()` and
  `ggml_backend_sycl_split_buffer_type()`.
- `src/llama.cpp`: `llama_default_buffer_type_split()` chooses CUDA/SYCL split
  buffers and needs the CPU-NUMA split-buffer branch.
- `src/llama-load-tensors.cpp`: `prepare_split_tensors()` creates shard tensor
  metadata and attaches it to the logical tensor.
- `src/llama-load-tensors.cpp`: split-mode graph setup prepares attention, FFN,
  MoE, and output tensor shard metadata after layer tensor creation.

Scheduler and execution expectations:

- `ggml_backend_sched_new()` receives one backend and compute buffer type per
  scheduler target.
- `ggml_backend_sched_backend_from_buffer()` and backend `supports_buft` decide
  tensor ownership from the buffer type.
- `ggml_backend_sched_split_graph()` treats `GGML_OP_REDUCE`, `GGML_OP_FAKE_CPY`,
  and marked nodes as split boundaries.
- Split-mode graph scheduling is enabled from `src/llama.cpp` through
  `ggml_backend_sched_set_split_mode_graph()` when the model is in
  `LLAMA_SPLIT_MODE_GRAPH`.
- CPU-NUMA shard matmuls should run on the backend that owns the shard buffer,
  with `ggml_cplan::numa_node` pinning worker threads to that node.

Reduction path finding:

- `ggml_reduce()` in `ggml/src/ggml.c` creates a `GGML_OP_REDUCE` node with
  `GGML_OP_ADD` in `op_params[0]`.
- Split graph builders emit this reduce for split attention, dense FFN, and MoE
  FFN paths in `src/llama-build-context.cpp`.
- CUDA supports `GGML_OP_REDUCE` through `ggml_cuda_op_reduce()` and advertises
  support in `ggml_backend_cuda_supports_op()`.
- CPU backend currently advertises default support for most ops, but CPU compute
  aborts on `GGML_OP_REDUCE` in `ggml/src/ggml.c`. This means CPU-NUMA cannot
  rely on the existing reduce node until CPU reduce-add is implemented or the
  scheduler routes reduce to a backend that can perform it correctly.

Decision:

CPU-NUMA should reuse the generic split tensor metadata, llama split tensor
creation, and scheduler split-mode graph machinery. It still needs a CPU-NUMA
split buffer type and a CPU reduce-add path. Do not remove the `--cpu-tp 2`
startup rejection until both are present and a deterministic correctness test
passes.

## Task B: CPU-NUMA Split Buffer Shape

Decision: implement a CPU-NUMA parent split buffer type that wraps existing
per-node `CPU-NUMA<i>` child buffer types. The parent buffer is a logical owner
for split tensors; node-local child buffers own the actual shard memory.

Internal representation:

- A singleton Linux-only parent buffer type named `CPU-NUMA-Split`.
- Parent buffer context stores no bulk allocation. It may keep lightweight
  accounting totals for debug logs, but shard buffers own their memory.
- Each logical tensor that has `tensor->extra` as `ggml_split_tensor_t` is
  initialized by allocating one child buffer for each present split shard.
- Child buffers use `ggml_backend_cpu_numa_buffer_type(i)` so existing
  NUMA-aware `supports_buft` dispatch maps shard tensors to `CPU-NUMA<i>`.
- The parent split buft reports `is_host == false`. This is intentional: llama
  split-mode graph currently skips split tensor preparation when the matrix
  buffer type is host.

Shard metadata:

- Shape metadata stays in `llama_split_tensor::tensor_splits` and
  `ggml_split_tensor_t::splits`, created by `prepare_split_tensors()`.
- No new shard list is needed in the parent buffer. The parent buffer inspects
  the logical tensor's existing `extra` pointer during `init_tensor`,
  `set_tensor`, and `get_tensor`.
- The split dimension and per-device shape are authoritative. Loading code must
  copy according to the shard shapes, not recompute split boundaries from
  `model.splits`.

Shard sizing and alignment:

- `get_alloc_size()` returns the sum of `ggml_nbytes(split)` for every present
  shard, rounded only where the child CPU-NUMA buffer allocator requires it.
- Child allocation uses the existing CPU tensor alignment. If later profiling
  shows a need for page alignment or huge pages, that should be a separate
  locality/performance task.
- For Task C, allocation may use the current `malloc`-backed CPU-NUMA child
  buffers. That proves representation and scheduler ownership. True node-local
  allocation or first-touch can be tightened after the split parent exists.

Loading behavior:

- `set_tensor()` must accept whole-tensor writes only: `offset == 0` and
  `size == ggml_nbytes(tensor)`.
- `split_dim < 0` means replicate the full logical tensor into every present
  shard.
- `split_dim == 0` copies row chunks along `ne[0]`.
- `split_dim == 1` copies contiguous chunks along `ne[1]`.
- `split_dim == 2` copies contiguous chunks along `ne[2]`.
- Explicit-range and repacked layouts can remain unsupported until Task D needs
  them; abort with a clear message rather than silently misloading.

Logical tensor exposure:

- The parent logical tensor gets a dummy non-null base pointer like CUDA split
  buffers. Runtime compute must use child shard tensors, not the parent `data`.
- Parent split buffers are only for weight/model tensors. Scheduler compute
  buffers for actual CPU-node execution remain per-node `CPU-NUMA<i>` buffers.
- Fallback CPU buffers keep their existing semantics and do not become split
  buffers.

Platform and failure modes:

- The parent split buffer API is Linux-only beside the existing CPU-NUMA child
  buffer APIs. Non-Linux builds should not expose it.
- `--cpu-tp 2` still rejects startup until allocation, loading, execution,
  reduce, and deterministic parity are validated.
- If fewer than two usable NUMA nodes are detected, CPU TP remains unavailable.
- If a split tensor references more shards than available CPU-NUMA child buffer
  types, allocation fails early.

## Validation Checkpoint 2026-05-25

Implemented since Task B:

- `--cpu-tp 2` now selects CPU-NUMA split-mode graph placement instead of
  rejecting startup.
- `CPU-NUMA-Split` allocates shard tensors on `CPU-NUMA0` and `CPU-NUMA1`
  child buffer types.
- `CPU-NUMA<i>` buffers now first-touch allocated pages while temporarily bound
  to the target NUMA node, so node buffer identity also influences physical page
  placement instead of being only scheduler metadata.
- CPU `GGML_OP_REDUCE` add is implemented for F32, F16, and BF16.
- CPU-TP graph construction uses the full reduced activation as the next shard
  input for CPU-TP, avoiding the GPU-oriented shortcut that reuses per-shard
  reduce sources.
- `tests/test-cpu-numa-tp.cpp` covers split-buffer set/get, CPU reduce-add,
  and scheduler reduce-add on NUMA machines.

Verified commands:

```bash
cmake --build build-debug-no-cuda --target llama-cli llama-server test-cpu-numa-tp -j4
ctest --test-dir build-debug-no-cuda -R test-cpu-numa-tp --output-on-failure
```

Both passed.

Tiny F32 model correctness:

```bash
build-debug-no-cuda/bin/llama-cli -m models/stories260K.gguf \
  -p "Hello" -n 8 -s 42 -t 1 -tb 1 -ngl 0 --no-warmup --temp 0 --no-display-prompt

build-debug-no-cuda/bin/llama-cli -m models/stories260K.gguf \
  -p "Hello" -n 8 -s 42 -t 1 -tb 1 --cpu-tp 2 --no-warmup --temp 0 --no-display-prompt
```

Both generated:

```text
 was a big, red ball
```

The same fixed-thread comparison for prompt `a` generated:

```text
 little girl named Lily went
```

Fixed `-t 1 -tb 1` is required for a stable token-level comparison on this
tiny model; the ordinary multithreaded CPU baseline can choose a different
argmax token.

Server startup smoke:

```bash
timeout 8s build-debug-no-cuda/bin/llama-server \
  -m models/stories260K.gguf --cpu-tp 2 -t 1 -tb 1 \
  --host 127.0.0.1 --port 18080 --no-warmup
```

The server loaded the model, initialized `CPU-NUMA0` and `CPU-NUMA1`, and
reached `HTTP server listening` on `127.0.0.1:18080`.

Quantized-model follow-up:

A larger quantized smoke model,
`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`
`tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`, was downloaded to
`models/perf/` for local validation. Baseline no-TP single-thread generation
worked:

```bash
build-debug-no-cuda/bin/llama-cli \
  -m models/perf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Once upon a time" -n 16 -s 42 -t 1 -tb 1 -ngl 0 \
  --no-warmup --temp 0 --no-display-prompt
```

Generated:

```text
, there was a young woman named Lily. Lily was a kind and
```

After fixing CPU-TP graph input handling, the same fixed-thread CPU-TP command
now generates the same text with default flash attention:

```bash
build-debug-no-cuda/bin/llama-cli \
  -m models/perf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Once upon a time" -n 16 -s 42 -t 1 -tb 1 --cpu-tp 2 \
  --no-warmup --temp 0 --no-display-prompt
```

Generated:

```text
, there was a young woman named Lily. Lily was a kind and
```

CPU-TP v1 has been narrowed to FFN-only split placement. Attention tensors,
rope tensors, and KV cache tensors stay on ordinary CPU/CPU-NUMA buffers, while
FFN and MoE tensors use `CPU-NUMA-Split`. This removes the split-attention flash
assertion class and allows both default flash attention and `--no-flash-attn`.
CPU-TP now enables split scheduler async by default because the FFN split graph
is stable with async and materially faster than the synchronous path.

Threaded local performance smoke on the same Q4 model is negative:

```bash
build-debug-no-cuda/bin/llama-cli \
  -m models/perf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Once upon a time in a small village near the mountains" \
  -n 64 -s 42 -t 52 -tb 52 -ngl 0 --no-warmup --temp 0 --no-display-prompt

build-debug-no-cuda/bin/llama-cli \
  -m models/perf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  -p "Once upon a time in a small village near the mountains" \
  -n 64 -s 42 -t 52 -tb 52 --cpu-tp 2 --no-warmup --temp 0 --no-display-prompt
```

Both generated the same text. After FFN-only placement and default async
scheduling, the pre-first-touch local sample measured about 219 prompt tokens/s
and 38.8 generation tokens/s for baseline CPU, versus about 47 prompt tokens/s
and 25.8 generation tokens/s for CPU-TP. After adding CPU-NUMA buffer
first-touch, a later sample measured about 148 prompt tokens/s and
42.4 generation tokens/s for baseline CPU, versus about 79 prompt tokens/s and
37.1 generation tokens/s for CPU-TP. The crash-prone full-attention split path
is gone and first-touch narrows the generation gap, but this is still not a
serving-performance win for this model/host configuration.

Server readiness on the Q4 model:

- The server had been running `common_speculative_is_compat(ctx)` during slot
  initialization even when no speculative decoding was requested. That helper
  decodes dummy token IDs and triggered
  `ggml/src/./iqk/fa/iqk_fa_templates.h:1157: GGML_ASSERT(S > 0)` for some
  thread counts.
- The speculative compatibility probe is now gated behind an actual
  speculative-decoding request.
- `llama-server --cpu-tp 2 -t 52 -tb 52 --no-warmup` reaches
  `HTTP server listening` and serves a `/completion` request for the Q4 model.
  With FFN-only placement and default async scheduling, the request generated
  the expected prefix and measured about 72 prompt tokens/s and 37 generation
  tokens/s after CPU-NUMA buffer first-touch.
- `numastat -p` on the live Q4 CPU-TP server showed process private memory
  distributed across both nodes after load, about 322 MiB on node 0 and
  443 MiB on node 1 in the local smoke.
- An uncommitted experiment that propagated the requested thread count into
  CPU-NUMA child backends made CPU-TP slower and could re-trigger the
  flash-attention assert, so it was not kept.
- After FFN-only placement, a second uncommitted experiment that assigned each
  CPU-NUMA child backend half of the requested threads improved prompt
  throughput but dropped Q4 generation to about 13.5 tokens/s, so it was also
  not kept.
- A later FFN-only/first-touch sweep of CPU-NUMA child backend thread counts
  showed the existing child backend default of 4 threads was the best generation
  point among 1, 2, 4, 8, and 16 threads. Eight threads improved prompt
  throughput but reduced generation throughput.

Representative Qwen 122B MoE gate:

The representative large-model gate used
`/mnt/storage/models/qwen3.5-122b-a10b-mtp-ud-q8_k_xl/UD-Q8_K_XL/Qwen3.5-122B-A10B-UD-Q8_K_XL-00001-of-00004.gguf`
with prompt `Hello`, seed 42, `-t 96 -tb 96 -c 2048 -b 128 -ub 128`,
`--no-warmup --temp 0 --no-display-prompt`.

Baseline ordinary CPU:

- Log: `/tmp/cpu-numa-tp-logs/qwen122b-q8-baseline-cpu-t96-n8.log`
- Output for `-n 8`: `, I am a 20 year`
- Load time: about 224 s
- Generation: about 3.43 tokens/s

CPU-TP after first-touch, before the split-norm fix:

- Log: `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-firsttouch-t96-n8.log`
- Output for `-n 8`: ` аген2 human232)0`
- Load time: about 382 s
- Generation: about 2.44 tokens/s
- Graph splits: 175
- `CPU-NUMA-Split` buffer: about 120420 MiB

Follow-up diagnostics did not remove the mismatch:

- `--graph-reduce-type f32`, `-n 4`:
  `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-firsttouch-grtf32-t96-n4.log`
  still generated ` аген2 human2` at about 2.12 tokens/s.
- `-no-fmoe -no-mmad`, `-n 4`:
  `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-firsttouch-no-fmoe-no-mmad-t96-n4.log`
  still generated ` аген2 human2` at about 2.05 tokens/s.
- `--no-offload-only-active-experts`, `-n 4`:
  `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-firsttouch-no-ooae-t96-n4.log`
  still generated ` аген2 human2` at about 2.09 tokens/s.
- A temporary no-forced-async build, `-n 4`:
  `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-no-forced-async-t96-n4.log`
  still generated ` аген2 human2` at about 1.51 tokens/s.

This rules out simple reduce precision, fused MoE up/gate, fused multi-add, and
active-expert scheduling as the immediate explanation. It also rules out the
forced CPU-TP async scheduler override as the correctness cause. The remaining
blocker is in the generic split MoE graph semantics or CPU-NUMA split tensor
execution for Qwen35MoE-shaped FFN/shared-expert layers. The code inspection
point to keep in mind is `llm_build_moe_ffn()` in
`src/llama-build-context.cpp`: routed experts and shared experts are split
across the FFN hidden dimension, not the expert axis, so global expert IDs are
still present on each shard. That makes a simple expert-index offset bug
unlikely.

Follow-up test coverage now includes byte-exact Q8_0 split-buffer roundtrips
for split dimensions 0, 1, and 2 in `tests/test-cpu-numa-tp.cpp`. That covers
the simple 3D quantized split tensor copy paths used by Qwen MoE weights and
passed. The same test file now also includes a scheduler-level Q8_0 MoE
hidden-split check: it runs two CPU-NUMA `mul_mat_id` up/down shards over the
FFN hidden dimension, reduces the shard outputs, and compares against a
separate plain-CPU full-weight Q8_0 reference. With 32-wide hidden shards this
passes within normal F32 accumulation-order tolerance, so the basic routed MoE
hidden-dimension split/reduce algebra is not the Qwen122B failure by itself.
It also includes a scheduler-level Q8_0 shared-expert/gate split check: two
CPU-NUMA shared-expert up/gate/down shards use a replicated scalar shared gate,
add the routed output to one shard before reduce, and compare against a
separate plain-CPU full-weight reference. That passes too, so the normal
shared-expert hidden split, scalar gate, add-once, and reduce algebra is not
the Qwen122B failure by itself either.

During that shared-expert test, leaving the shard-local SwiGLU/gate multiply as
`ggml_fused_mul_unary` exposed intermittent large synthetic mismatches under
the CPU-NUMA scheduler. The CPU-TP graph builder now avoids
`ggml_fused_mul_unary` when `cpu_tp > 1`, spelling those paths as explicit
unary plus multiply. The synthetic shared-expert check passed 40 consecutive
runs with this unfused CPU-TP path.

A representative Qwen122B CPU-TP `-n 4` rerun with that unfused CPU-TP path
still generated ` аген2 human2`:

```text
/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-no-fused-mul-unary-cputp-t96-n4.log
load time 293291.88 ms, eval time 1916.81 ms / 4 tokens, 2.09 tok/s
```

So fused unary multiply instability is a real CPU-TP graph hazard to avoid, but
it is not the representative Qwen35MoE correctness cause by itself.

The actual representative correctness bug was in the fully split routed MoE FFN
path. `do_split_norm()` only applied the per-shard norm when the norm tensor was
itself split and carried split metadata. Qwen35MoE `ffn_norm` tensors were being
kept unsplit, so the split routed FFN path skipped `ffn_norm` entirely. Applying
the norm tensor even when it is unsplit restored the representative output, but
the first fix made the graph slow because the unsplit norm forced extra
cross-node copies.

The follow-up loader fix routes `.ffn_norm.` tensors to the CPU-TP split context
when `--cpu-tp 2` is active. That lets the existing split tensor loading path
own and replicate those norms like the other split FFN tensors.

Representative Qwen122B CPU-TP after the split-norm and split-context fixes:

- Log:
  `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-split-norm-splitctx-t96-n4-pipefail.log`
- Output for `-n 4`: `, I am a`
- Load time: about 391 s
- Generation: about 2.09 tokens/s
- Graph splits: 200

This matches the ordinary CPU baseline prefix for the same prompt and seed.
The earlier wrong output, ` аген2 human2`, is no longer reproduced by the
representative short gate.

Follow-up CPU-NUMA child backend thread tuning:

- Temporarily forcing higher CPU-NUMA child backend thread counts on the dense
  Q4 TinyLlama smoke was harmful. With `LLAMA_CPU_TP_NODE_THREADS=8` and `16`
  in an experiment build, the Q4 CPU-TP generation rate dropped to about
  19.7-19.9 tokens/s, well below the prior first-touch CPU-TP Q4 point of
  about 37.1 tokens/s.
- The same temporary thread override on the representative Qwen35MoE 122B Q8
  MoE target improved generation. At 16 child backend threads, the `-n 4`
  short gate generated `, I am a` and measured about 3.30 tokens/s. At 24 child
  backend threads it regressed to about 2.99 tokens/s.
- The initial kept heuristic was MoE-only: for `--cpu-tp 2` models with
  experts, CPU-NUMA child backends used `clamp(n_threads / 6, 4, 16)` threads.
  Dense models kept the previous default child backend threading.
- The OpenClaw rerun showed CPU-TP generation close to but still below the
  clean `origin/main` baseline, while core utilization looked lower than the
  later ordinary CPU run. Decision: keep the heuristic as the default because
  previous dense Q4 trials regressed badly when child backend threads were
  raised globally, but expose explicit trial knobs:
  `--cpu-tp-threads N` for generation and `--cpu-tp-threads-batch N` for
  prompt/batch processing. After observing that the host is mainly dedicated to
  large model serving, the default was changed to divide total graph threads by
  CPU-TP node count: with `--cpu-tp 2 -t 96 -tb 96`, each NUMA child backend
  gets 48 threads. `0` means that default division; if only
  `--cpu-tp-threads` is set, batch CPU-TP child backends use that override too.
- The rebuilt-source Qwen35MoE 122B Q8 short gate with this heuristic generated
  `, I am a`, loaded in about 382 s, and measured about 3.60 tokens/s:

```text
/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-moe-thread-heuristic-t96-n4-pipefail.log
load time 381770.25 ms, eval time 1111.23 ms / 4 tokens, 3.60 tok/s
```

The apples-to-apples `-n 8` rerun with the same heuristic generated the same
prefix as the ordinary CPU baseline, `, I am a 20 year`, loaded in about 390 s,
and measured about 4.21 tokens/s:

```text
/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-moe-thread-heuristic-t96-n8-pipefail.log
load time 390156.12 ms, eval time 1899.36 ms / 8 tokens, 4.21 tok/s
```

That is about a 23% generation-throughput improvement over the ordinary CPU
baseline `-n 8` point of about 3.43 tokens/s, so the representative Task H
throughput gate is now met. Load time remains materially worse than the
ordinary CPU baseline because the CPU-NUMA split-buffer path still allocates and
copies about 120420 MiB during load.

The initial split-buffer load path first-touched every CPU-NUMA shard with a
zero write during allocation, then copied the real tensor bytes during
`ggml_backend_tensor_set()`. For Qwen this doubled the memory traffic over the
120420 MiB split buffer. CPU-NUMA split-buffer shards now allocate without the
separate zero pass; the real tensor copy runs while pinned to the destination
NUMA node, so the copy itself establishes page placement.

Representative Qwen122B CPU-TP after copy-as-first-touch loading:

- `-n 1` log:
  `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-copy-firsttouch-t96-n1-pipefail.log`
  loaded in about 236 s and generated `,`.
- `-n 8` log:
  `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-copy-firsttouch-t96-n8-pipefail.log`
  generated `, I am a 20 year`, loaded in about 276 s, and measured about
  4.34 tokens/s.

The `-n 8` result is about 27% faster than the ordinary CPU generation baseline
and only about 52 s slower to load in this local sample. This makes the
representative CLI gate much closer to adoption-ready, but the server path
still needs a deliberate validation run before changing serving config.

Representative Qwen122B CPU-TP server validation:

- Command shape:
  `llama-server -m <Qwen122B Q8 split GGUF> -t 96 -tb 96 -c 2048 -b 128 -ub 128 --no-warmup --temp 0 --cpu-tp 2 --host 127.0.0.1 --port 18080 --log-format text`
- Server log:
  `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-server-copy-firsttouch-t96-p18080.log`
- Completion response:
  `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-server-copy-firsttouch-completion-response.json`
- Build: `fd122715`
- Startup: model load began at log timestamp `1779720398`; `model loaded` and
  `HTTP server listening` were logged at `1779720625`, about 227 s later.
- Request: `POST /completion` with prompt `Hello`, `n_predict: 8`,
  `temperature: 0`, `seed: 42`, and `cache_prompt: false`.
- Response content: `, I am a 20 year`
- Server timing: prompt eval 340.12 ms / 1 token, eval 1540.46 ms / 8 tokens,
  about 5.19 tokens/s.

This validates the representative Qwen server path without changing Docker or
the active serving config.

Representative Qwen122B CPU-TP OpenClaw payload smoke:

- Command shape:
  `llama-server -m <Qwen122B Q8 split GGUF> -t 96 -tb 96 -c 32768 -b 128 -ub 128 --no-warmup --temp 0.6 --cpu-tp 2 --host 127.0.0.1 --port 18080 --log-format text`
- Payload command:
  `HOST=127.0.0.1 PORT=18080 MAX_TOKENS=64 TEMPERATURE=0.6 OUT_DIR=/tmp/cpu-numa-tp-openclaw-smoke-rmscast-addfix scripts/test_openclaw_payload_local.sh`
- Server log:
  `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-server-openclaw-c32768-rmscast-addfix-p18080.log`
- Payload output:
  `/tmp/cpu-numa-tp-openclaw-smoke-rmscast-addfix/payload_local_20260525_204559`
- Result summary: HTTP 200, no error, choices present, finish reason `stop`.
- Token counts: 26,747 prompt tokens, 64 generated tokens.
- Server timing: prompt eval 1,020,093.53 ms / 26,747 tokens, about
  26.22 prompt tokens/s; eval 71,834.53 ms / 64 tokens, about
  0.89 generated tokens/s; total 1,091,928.06 ms / 26,811 tokens.
- Clean rerun with the active Docker server and OpenClaw gateway stopped:
  - Branch server log:
    `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-server-openclaw-c32768-rerun2-branch-p18080.log`
  - Branch payload output:
    `/tmp/cpu-numa-tp-openclaw-smoke-rerun2-branch/payload_local_20260525_212340`
  - Branch result: HTTP 200, 26,747 prompt tokens, 64 generated tokens,
    34.55 prompt tokens/s, 5.13 generated tokens/s.
  - `origin/main` baseline was built in `/tmp/ik_llama_openclaw_main` at
    `b4e1d916` and run with the same command shape minus `--cpu-tp 2`.
  - Main server log:
    `/tmp/cpu-numa-tp-logs/qwen122b-q8-main-server-openclaw-c32768-rerun2-p18080.log`
  - Main payload output:
    `/tmp/cpu-numa-tp-openclaw-smoke-rerun2-main/payload_local_20260525_213954`
  - Main result: HTTP 200, 26,747 prompt tokens, 64 generated tokens,
    65.80 prompt tokens/s, 5.74 generated tokens/s.
  - Caveat: `origin/main` ignored the MTP-layer tensors from this GGUF, so this
    is a practical pre-branch baseline, not a same-feature MTP/CPU-TP comparison.
- CPU-TP child-thread 48/48 trial:
  - Command shape:
    `llama-server -m <Qwen122B Q8 split GGUF> -t 96 -tb 96 -c 32768 -b 128 -ub 128 --no-warmup --temp 0.6 --cpu-tp 2 --cpu-tp-threads 48 --cpu-tp-threads-batch 48 --host 127.0.0.1 --port 18080 --log-format text`
  - Server log:
    `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-server-openclaw-c32768-cputpthreads48-p18080.log`
  - Payload output:
    `/tmp/cpu-numa-tp-openclaw-smoke-cputpthreads48/payload_local_20260525_224727`
  - Result: HTTP 200, 26,747 prompt tokens, 64 generated tokens,
    44.62 prompt tokens/s, 4.12 generated tokens/s.
  - This improved prompt throughput over the prior clean CPU-TP run
    (34.55 prompt tokens/s) but regressed generation versus the prior CPU-TP
    run (5.13 generated tokens/s), and still trailed clean `origin/main` on
    both metrics.
  - Operator observation during the 48/48 run: threads were broadly loaded, but
    per-core utilization appeared around 50-75% and CPU package power around
    150 W of a 205 W envelope. Treat the resulting 20-25% possible uplift as a
    theoretical headroom target, not a measured expectation; likely blockers
    include synchronization, memory bandwidth, NUMA locality, and shard
    scheduling overhead.
  - Memory check on the live process showed about 128.5 GiB RSS, not 2x model
    memory. `numastat -p` showed private memory split across both nodes, about
    63.4 GiB on node 0 and 67.4 GiB on node 1. This is expected: CPU-TP is
    tensor-parallel sharding, not model replication. The TG deficit is therefore
    more likely bandwidth/scheduling limited than caused by missing replicated
    weights. This run also did not use speculative/MTP generation, so TG was
    ordinary single-token generation.
- CPU-TP child-thread 16/48 trial with a 1024-token generation window:
  - Command shape:
    `llama-server -m <Qwen122B Q8 split GGUF> -t 96 -tb 96 -c 32768 -b 128 -ub 128 --no-warmup --temp 0.6 --cpu-tp 2 --cpu-tp-threads 16 --cpu-tp-threads-batch 48 --host 127.0.0.1 --port 18080 --log-format text`
  - Server log:
    `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-server-openclaw-c32768-cputpthreads16-48-maxtok1024-p18080.log`
  - Payload output:
    `/tmp/cpu-numa-tp-openclaw-smoke-cputpthreads16-48-maxtok1024/payload_local_20260525_230542`
  - Result: HTTP 200, 26,747 prompt tokens, 1,024 generated tokens,
    45.67 prompt tokens/s, 5.41 generated tokens/s.
  - This keeps the PP gain from higher batch child threads and recovers TG
    close to the clean `origin/main` 64-token TG sample. The longer 1,024-token
    window is a more meaningful TG measurement than the earlier 64-token smoke.
  - Interpretation: for non-MTP serving, lower generation child threads and
    higher prompt/batch child threads are likely a better default shape than
    `96/96` everywhere. MTP is not changed by these flags, but it should be
    tested separately because MTP decode is more batch-like and may prefer the
    runner's higher `-t/-tb` shape.
- Runner-shaped MTP comparison:
  - Shared serving shape: `-t 96 -tb 96 -c 32768 -b 1024 -ub 1024 --temp 0.6
    --top-p 0.95 --top-k 20 --min-p 0 --multi-token-prediction --spec-type
    mtp --draft-n 8 --draft-min 0 --draft-p-min 0.8`, 1,024 generated token
    cap. This uses the checked-in Qwen3.5 122B MTP runner batch/thread shape,
    not the earlier `128/128` smoke batch.
  - Clean `origin/main` MTP baseline:
    `/tmp/cpu-numa-tp-logs/qwen122b-q8-main-mtp-server-openclaw-c32768-b1024-maxtok1024-p18080.log`,
    output
    `/tmp/cpu-numa-tp-openclaw-smoke-main-mtp-b1024-maxtok1024/payload_local_20260525_234439`.
    Result: HTTP 200, 26,747 prompt tokens, 1,024 generated tokens,
    121.69 prompt tokens/s, 5.82 generated tokens/s, draft acceptance 0.778.
  - CPU-TP MTP 48/48 child-thread run:
    `/tmp/cpu-numa-tp-logs/qwen122b-q8-cputp-mtp-server-openclaw-c32768-b1024-cputpthreads48-48-maxtok1024-p18080.log`,
    output
    `/tmp/cpu-numa-tp-openclaw-smoke-cputp-mtp-b1024-cputpthreads48-48-maxtok1024/payload_local_20260525_235403`.
    Result: HTTP 200, 26,747 prompt tokens, 1,024 generated tokens,
    50.57 prompt tokens/s, 4.78 generated tokens/s, draft acceptance 0.824.
  - Interpretation: MTP itself works on CPU-TP, and acceptance was not the
    problem. CPU-TP still trails heavily on PP and also trails TG in this
    MTP/48/48 shape. The PP deficit remains the primary blocker.
- Task H outcome: this proves the CPU-TP long-prompt server path is functional,
  but it does not beat the clean `origin/main` baseline on this payload. Keep
  the branch as validated CPU-NUMA TP work, but do not deploy it as the active
  serving config from current performance evidence.
- This smoke exposed two long-prefill CPU graph gaps that short prompts did
  not hit:
  - RMS norm inputs can arrive as reduced half tensors; `llm_build_norm()` now
    casts RMS inputs to F32 before constructing the norm op.
  - CPU `ggml_add` now supports F32 plus F16/BF16 right-hand operands with
    normal `ggml_can_repeat()` broadcasting, covered by
    `tests/test-cpu-numa-tp.cpp`.

Dense Q4 guard after the split-buffer load change:

- Log: `/tmp/cpu-numa-tp-logs/cputp-q4-copy-firsttouch-t52-n16.log`
- Output prefix remained `, there lived a young girl named Lily. Lily was a kind and`
- Load time: about 1.7 s
- Generation: about 34.0 tokens/s on the short `-n 16` smoke

These checks passed under:

```bash
cmake --build build-debug-no-cuda --target llama-cli llama-server test-cpu-numa-tp -j4
ctest --test-dir build-debug-no-cuda -R test-cpu-numa-tp --output-on-failure
git diff --check
```

Current verdict:

CPU-NUMA tensor parallelism has narrow CLI correctness evidence for F32 and Q4
smoke models, including `--no-flash-attn`; the Q4 server path can start and
answer a request; and the representative Qwen35MoE 122B Q8 server path now
starts and serves the baseline-matching short completion. The representative
Qwen35MoE 122B Q8 short gate matches the ordinary CPU baseline prefix after
fixing split FFN norm handling. A MoE-only CPU-NUMA child thread heuristic
clears the representative Task H generation-throughput target on the
apples-to-apples `-n 8` CLI gate, copy-as-first-touch split loading narrows
the large load-time penalty substantially, and the full local OpenClaw payload
smoke now completes through long prefill and generation. The clean rerun after
stopping unrelated services improved CPU-TP OpenClaw generation from 0.89 to
5.13 tokens/s, but the clean `origin/main` baseline still measured 5.74
tokens/s on the same payload shape. This branch is validated for CPU-NUMA TP
functionality but should not be adopted as the active serving config on current
performance evidence. A 48/48 child-thread trial improved prompt throughput but
regressed generation throughput, so the next optimization pass should focus on
theoretical headroom from synchronization, memory bandwidth, NUMA locality, and
shard scheduling rather than only increasing worker counts. No active serving
config has been changed in this spike.
