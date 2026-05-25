#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok, const char * msg) {
    if (!ok) {
        std::fprintf(stderr, "%s\n", msg);
        std::exit(1);
    }
}

static void require_near(float got, float want, const char * msg) {
    if (std::fabs(got - want) > 1e-5f) {
        std::fprintf(stderr, "%s: got %.8f, want %.8f\n", msg, got, want);
        std::exit(1);
    }
}

static void roundtrip_split(int split_dim) {
#ifndef __gnu_linux__
    (void) split_dim;
#else
    const int64_t ne0 = 4;
    const int64_t ne1 = 3;
    const int64_t ne2 = 2;

    ggml_init_params params = {
        /* .mem_size   = */ 32*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    ggml_tensor * parent = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_set_name(parent, "cpu_numa_split_test");

    ggml_tensor * shards[2] = {};
    if (split_dim == 0) {
        shards[0] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, ne1, ne2);
        shards[1] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, ne1, ne2);
    } else if (split_dim == 1) {
        shards[0] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, 1, ne2);
        shards[1] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, 2, ne2);
    } else if (split_dim == 2) {
        shards[0] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, 1);
        shards[1] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, 1);
    } else {
        require(false, "unsupported split dimension");
    }

    ggml_split_tensor_t split = {
        /* .n_device  = */ 2,
        /* .split_dim = */ split_dim,
        /* .tensor    = */ parent,
        /* .splits    = */ shards,
    };
    parent->extra = &split;

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_numa_split_buffer_type();
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, parent));
    require(buffer != nullptr, "failed to allocate CPU-NUMA split buffer");

    parent->buffer = buffer;
    parent->data = ggml_backend_buffer_get_base(buffer);
    ggml_backend_buffer_init_tensor(buffer, parent);

    std::vector<float> in(ggml_nelements(parent));
    for (size_t i = 0; i < in.size(); ++i) {
        in[i] = (float) i + 0.25f;
    }

    std::vector<float> out(in.size(), 0.0f);
    ggml_backend_tensor_set(parent, in.data(), 0, ggml_nbytes(parent));
    ggml_backend_tensor_get(parent, out.data(), 0, ggml_nbytes(parent));

    for (size_t i = 0; i < in.size(); ++i) {
        require_near(out[i], in[i], "split-buffer roundtrip mismatch");
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
#endif
}

static void test_reduce_add_cpu_backend() {
    ggml_init_params params = {
        /* .mem_size   = */ 1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * srcs[2] = { a, b };
    ggml_tensor * out = ggml_reduce(ctx, srcs, 2, GGML_OP_ADD);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr, "failed to initialize CPU backend");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate CPU backend tensors");

    const float av[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float bv[4] = { 10.0f, 20.0f, 30.0f, 40.0f };
    ggml_backend_tensor_set(a, av, 0, sizeof(av));
    ggml_backend_tensor_set(b, bv, 0, sizeof(bv));

    require(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS, "CPU reduce-add graph failed");

    float got[4] = {};
    ggml_backend_tensor_get(out, got, 0, sizeof(got));
    for (int i = 0; i < 4; ++i) {
        require_near(got[i], av[i] + bv[i], "CPU reduce-add mismatch");
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

static void test_scheduler_reduce_on_numa_backends() {
#ifndef __gnu_linux__
    std::puts("scheduler NUMA test skipped: non-Linux platform");
#else
    if (ggml_numa_node_count() == 0) {
        ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    }
    if (!ggml_is_numa() || ggml_numa_node_count() < 2) {
        std::puts("scheduler NUMA test skipped: fewer than two NUMA nodes");
        return;
    }

    ggml_init_params params = {
        /* .mem_size   = */ 1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    ggml_tensor * a0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor * b0 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor * a1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor * b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    ggml_tensor * p0 = ggml_mul(ctx, a0, b0);
    ggml_tensor * p1 = ggml_mul(ctx, a1, b1);
    ggml_tensor * srcs[2] = { p0, p1 };
    ggml_tensor * out = ggml_reduce(ctx, srcs, 2, GGML_OP_ADD);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_t backends[3] = {
        ggml_backend_cpu_init_with_numa(0),
        ggml_backend_cpu_init_with_numa(1),
        ggml_backend_cpu_init(),
    };
    require(backends[0] && backends[1] && backends[2], "failed to initialize CPU-NUMA backends");

    ggml_backend_buffer_type_t bufts[3] = {
        ggml_backend_cpu_numa_buffer_type(0),
        ggml_backend_cpu_numa_buffer_type(1),
        ggml_backend_get_default_buffer_type(backends[2]),
    };

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 3, GGML_DEFAULT_GRAPH_SIZE, true);
    require(sched != nullptr, "failed to initialize scheduler");
    ggml_backend_sched_set_split_mode_graph(sched, true, false);
    ggml_backend_sched_set_tensor_backend(sched, p0, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, p1, backends[1]);
    require(ggml_backend_sched_alloc_graph(sched, gf), "failed to allocate scheduler graph");

    const float a0v[3] = { 1.0f, 2.0f, 3.0f };
    const float b0v[3] = { 10.0f, 20.0f, 30.0f };
    const float a1v[3] = { 4.0f, 5.0f, 6.0f };
    const float b1v[3] = { 100.0f, 200.0f, 300.0f };
    ggml_backend_tensor_set(a0, a0v, 0, sizeof(a0v));
    ggml_backend_tensor_set(b0, b0v, 0, sizeof(b0v));
    ggml_backend_tensor_set(a1, a1v, 0, sizeof(a1v));
    ggml_backend_tensor_set(b1, b1v, 0, sizeof(b1v));

    require(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS, "scheduler reduce-add graph failed");

    float got[3] = {};
    ggml_backend_tensor_get(out, got, 0, sizeof(got));
    for (int i = 0; i < 3; ++i) {
        require_near(got[i], a0v[i]*b0v[i] + a1v[i]*b1v[i], "scheduler reduce-add mismatch");
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(backends[0]);
    ggml_backend_free(backends[1]);
    ggml_backend_free(backends[2]);
    ggml_free(ctx);
#endif
}

int main() {
    roundtrip_split(0);
    roundtrip_split(1);
    roundtrip_split(2);
    test_reduce_add_cpu_backend();
    test_scheduler_reduce_on_numa_backends();
    std::puts("cpu-numa-tp tests passed");
    return 0;
}
