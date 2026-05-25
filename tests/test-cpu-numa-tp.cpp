#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

static void require_near_tol(float got, float want, float tol, const char * msg) {
    const float scale = std::max(1.0f, std::fabs(want));
    if (std::fabs(got - want) > tol*scale) {
        std::fprintf(stderr, "%s: got %.8f, want %.8f, tol %.8f\n", msg, got, want, tol);
        std::exit(1);
    }
}

static size_t idx3(int64_t i0, int64_t i1, int64_t i2, int64_t ne0, int64_t ne1) {
    return (size_t) (i0 + i1*ne0 + i2*ne0*ne1);
}

static size_t idx2(int64_t i0, int64_t i1, int64_t ne0) {
    return (size_t) (i0 + i1*ne0);
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

static void roundtrip_split_bytes(ggml_type type, int split_dim) {
#ifndef __gnu_linux__
    (void) type;
    (void) split_dim;
#else
    const int64_t ne0 = 64;
    const int64_t ne1 = 3;
    const int64_t ne2 = 2;

    ggml_init_params params = {
        /* .mem_size   = */ 32*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    ggml_tensor * parent = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    ggml_set_name(parent, "cpu_numa_split_bytes_test");

    ggml_tensor * shards[2] = {};
    if (split_dim == 0) {
        shards[0] = ggml_new_tensor_3d(ctx, type, 32, ne1, ne2);
        shards[1] = ggml_new_tensor_3d(ctx, type, 32, ne1, ne2);
    } else if (split_dim == 1) {
        shards[0] = ggml_new_tensor_3d(ctx, type, ne0, 1, ne2);
        shards[1] = ggml_new_tensor_3d(ctx, type, ne0, 2, ne2);
    } else if (split_dim == 2) {
        shards[0] = ggml_new_tensor_3d(ctx, type, ne0, ne1, 1);
        shards[1] = ggml_new_tensor_3d(ctx, type, ne0, ne1, 1);
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
    require(buffer != nullptr, "failed to allocate CPU-NUMA split byte buffer");

    parent->buffer = buffer;
    parent->data = ggml_backend_buffer_get_base(buffer);
    ggml_backend_buffer_init_tensor(buffer, parent);

    std::vector<uint8_t> in(ggml_nbytes(parent));
    for (size_t i = 0; i < in.size(); ++i) {
        in[i] = (uint8_t) ((i * 37u + 13u) & 0xffu);
    }

    std::vector<uint8_t> out(in.size(), 0);
    ggml_backend_tensor_set(parent, in.data(), 0, ggml_nbytes(parent));
    ggml_backend_tensor_get(parent, out.data(), 0, ggml_nbytes(parent));

    require(std::memcmp(out.data(), in.data(), in.size()) == 0, "split-buffer byte roundtrip mismatch");

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

static void test_scheduler_moe_hidden_split() {
#ifndef __gnu_linux__
    std::puts("scheduler MoE split test skipped: non-Linux platform");
#else
    if (ggml_numa_node_count() == 0) {
        ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    }
    if (!ggml_is_numa() || ggml_numa_node_count() < 2) {
        std::puts("scheduler MoE split test skipped: fewer than two NUMA nodes");
        return;
    }

    constexpr int64_t n_embd        = 32;
    constexpr int64_t n_ff0         = 32;
    constexpr int64_t n_ff1         = 32;
    constexpr int64_t n_ff          = n_ff0 + n_ff1;
    constexpr int64_t n_expert      = 3;
    constexpr int64_t n_expert_used = 2;
    constexpr int64_t n_tokens      = 2;
    constexpr ggml_type weight_type = GGML_TYPE_Q8_0;

    ggml_init_params params = {
        /* .mem_size   = */ 1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    ggml_tensor * x       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_expert_used, n_tokens);
    ggml_tensor * ids     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);
    ggml_tensor * up0_w   = ggml_new_tensor_3d(ctx, weight_type, n_embd, n_ff0, n_expert);
    ggml_tensor * up1_w   = ggml_new_tensor_3d(ctx, weight_type, n_embd, n_ff1, n_expert);
    ggml_tensor * down0_w = ggml_new_tensor_3d(ctx, weight_type, n_ff0, n_embd, n_expert);
    ggml_tensor * down1_w = ggml_new_tensor_3d(ctx, weight_type, n_ff1, n_embd, n_expert);

    ggml_tensor * up0   = ggml_mul_mat_id(ctx, up0_w, x, ids);
    ggml_tensor * down0 = ggml_mul_mat_id(ctx, down0_w, up0, ids);
    ggml_tensor * up1   = ggml_mul_mat_id(ctx, up1_w, x, ids);
    ggml_tensor * down1 = ggml_mul_mat_id(ctx, down1_w, up1, ids);
    ggml_tensor * srcs[2] = { down0, down1 };
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
    ggml_backend_sched_set_tensor_backend(sched, up0_w,   backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, down0_w, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, up0,     backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, down0,   backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, up1_w,   backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, down1_w, backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, up1,     backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, down1,   backends[1]);
    require(ggml_backend_sched_alloc_graph(sched, gf), "failed to allocate scheduler MoE split graph");

    std::vector<float> xv(ggml_nelements(x));
    for (size_t i = 0; i < xv.size(); ++i) {
        xv[i] = 0.1f + 0.03f*(float) i;
    }
    const int32_t idv[n_expert_used*n_tokens] = {
        0, 2,
        1, 0,
    };

    std::vector<float> up0v(ggml_nelements(up0_w));
    std::vector<float> up1v(ggml_nelements(up1_w));
    std::vector<float> down0v(ggml_nelements(down0_w));
    std::vector<float> down1v(ggml_nelements(down1_w));
    std::vector<float> upv((size_t) n_embd*n_ff*n_expert);
    std::vector<float> downv((size_t) n_ff*n_embd*n_expert);
    for (size_t i = 0; i < up0v.size(); ++i)   up0v[i]   = 0.02f + 0.001f*(float) i;
    for (size_t i = 0; i < up1v.size(); ++i)   up1v[i]   = 0.03f + 0.002f*(float) i;
    for (size_t i = 0; i < down0v.size(); ++i) down0v[i] = 0.04f + 0.003f*(float) i;
    for (size_t i = 0; i < down1v.size(); ++i) down1v[i] = 0.05f + 0.004f*(float) i;

    for (int64_t ie = 0; ie < n_expert; ++ie) {
        for (int64_t ih = 0; ih < n_ff0; ++ih) {
            for (int64_t ic = 0; ic < n_embd; ++ic) {
                upv[idx3(ic, ih, ie, n_embd, n_ff)] = up0v[idx3(ic, ih, ie, n_embd, n_ff0)];
            }
        }
        for (int64_t ih = 0; ih < n_ff1; ++ih) {
            for (int64_t ic = 0; ic < n_embd; ++ic) {
                upv[idx3(ic, n_ff0 + ih, ie, n_embd, n_ff)] = up1v[idx3(ic, ih, ie, n_embd, n_ff1)];
            }
        }
        for (int64_t io = 0; io < n_embd; ++io) {
            for (int64_t ih = 0; ih < n_ff0; ++ih) {
                downv[idx3(ih, io, ie, n_ff, n_embd)] = down0v[idx3(ih, io, ie, n_ff0, n_embd)];
            }
            for (int64_t ih = 0; ih < n_ff1; ++ih) {
                downv[idx3(n_ff0 + ih, io, ie, n_ff, n_embd)] = down1v[idx3(ih, io, ie, n_ff1, n_embd)];
            }
        }
    }

    std::vector<uint8_t> up0q(ggml_nbytes(up0_w));
    std::vector<uint8_t> up1q(ggml_nbytes(up1_w));
    std::vector<uint8_t> down0q(ggml_nbytes(down0_w));
    std::vector<uint8_t> down1q(ggml_nbytes(down1_w));
    ggml_quantize_chunk(weight_type, up0v.data(),   up0q.data(),   0, ggml_nrows(up0_w),   up0_w->ne[0],   nullptr, nullptr);
    ggml_quantize_chunk(weight_type, up1v.data(),   up1q.data(),   0, ggml_nrows(up1_w),   up1_w->ne[0],   nullptr, nullptr);
    ggml_quantize_chunk(weight_type, down0v.data(), down0q.data(), 0, ggml_nrows(down0_w), down0_w->ne[0], nullptr, nullptr);
    ggml_quantize_chunk(weight_type, down1v.data(), down1q.data(), 0, ggml_nrows(down1_w), down1_w->ne[0], nullptr, nullptr);

    ggml_backend_tensor_set(x,       xv.data(),     0, ggml_nbytes(x));
    ggml_backend_tensor_set(ids,     idv,           0, ggml_nbytes(ids));
    ggml_backend_tensor_set(up0_w,   up0q.data(),   0, ggml_nbytes(up0_w));
    ggml_backend_tensor_set(up1_w,   up1q.data(),   0, ggml_nbytes(up1_w));
    ggml_backend_tensor_set(down0_w, down0q.data(), 0, ggml_nbytes(down0_w));
    ggml_backend_tensor_set(down1_w, down1q.data(), 0, ggml_nbytes(down1_w));

    require(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS, "scheduler MoE split graph failed");

    std::vector<float> got(ggml_nelements(out), 0.0f);
    ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));

    ggml_context * ctx_ref = ggml_init(params);
    require(ctx_ref != nullptr, "failed to initialize reference ggml context");
    ggml_tensor * x_ref     = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, n_embd, n_expert_used, n_tokens);
    ggml_tensor * ids_ref   = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_I32, n_expert_used, n_tokens);
    ggml_tensor * up_w_ref  = ggml_new_tensor_3d(ctx_ref, weight_type, n_embd, n_ff, n_expert);
    ggml_tensor * down_w_ref = ggml_new_tensor_3d(ctx_ref, weight_type, n_ff, n_embd, n_expert);
    ggml_tensor * up_full   = ggml_mul_mat_id(ctx_ref, up_w_ref, x_ref, ids_ref);
    ggml_tensor * ref_out   = ggml_mul_mat_id(ctx_ref, down_w_ref, up_full, ids_ref);
    ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(gf_ref, ref_out);

    ggml_backend_t backend_ref = ggml_backend_cpu_init();
    require(backend_ref != nullptr, "failed to initialize reference CPU backend");
    ggml_backend_buffer_t buffer_ref = ggml_backend_alloc_ctx_tensors(ctx_ref, backend_ref);
    require(buffer_ref != nullptr, "failed to allocate reference CPU tensors");
    std::vector<uint8_t> upq(ggml_nbytes(up_w_ref));
    std::vector<uint8_t> downq(ggml_nbytes(down_w_ref));
    ggml_quantize_chunk(weight_type, upv.data(),   upq.data(),   0, ggml_nrows(up_w_ref),   up_w_ref->ne[0],   nullptr, nullptr);
    ggml_quantize_chunk(weight_type, downv.data(), downq.data(), 0, ggml_nrows(down_w_ref), down_w_ref->ne[0], nullptr, nullptr);

    ggml_backend_tensor_set(x_ref,      xv.data(),   0, ggml_nbytes(x_ref));
    ggml_backend_tensor_set(ids_ref,    idv,         0, ggml_nbytes(ids_ref));
    ggml_backend_tensor_set(up_w_ref,   upq.data(),  0, ggml_nbytes(up_w_ref));
    ggml_backend_tensor_set(down_w_ref, downq.data(), 0, ggml_nbytes(down_w_ref));
    require(ggml_backend_graph_compute(backend_ref, gf_ref) == GGML_STATUS_SUCCESS, "reference MoE graph failed");

    std::vector<float> want(ggml_nelements(ref_out), 0.0f);
    ggml_backend_tensor_get(ref_out, want.data(), 0, ggml_nbytes(ref_out));

    for (size_t i = 0; i < got.size(); ++i) {
        require_near_tol(got[i], want[i], 1e-5f, "scheduler MoE hidden-split mismatch");
    }

    ggml_backend_buffer_free(buffer_ref);
    ggml_backend_free(backend_ref);
    ggml_free(ctx_ref);

    ggml_backend_sched_free(sched);
    ggml_backend_free(backends[0]);
    ggml_backend_free(backends[1]);
    ggml_backend_free(backends[2]);
    ggml_free(ctx);
#endif
}

static void test_scheduler_shared_expert_gate_split() {
#ifndef __gnu_linux__
    std::puts("scheduler shared-expert split test skipped: non-Linux platform");
#else
    if (ggml_numa_node_count() == 0) {
        ggml_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    }
    if (!ggml_is_numa() || ggml_numa_node_count() < 2) {
        std::puts("scheduler shared-expert split test skipped: fewer than two NUMA nodes");
        return;
    }

    constexpr int64_t n_embd   = 32;
    constexpr int64_t n_ff0    = 32;
    constexpr int64_t n_ff1    = 32;
    constexpr int64_t n_ff     = n_ff0 + n_ff1;
    constexpr int64_t n_tokens = 3;
    constexpr ggml_type weight_type = GGML_TYPE_Q8_0;

    ggml_init_params params = {
        /* .mem_size   = */ 1024*1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to initialize ggml context");

    ggml_tensor * x          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * routed_w   = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_embd);
    ggml_tensor * gate0_w    = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_ff0);
    ggml_tensor * gate1_w    = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_ff1);
    ggml_tensor * up0_w      = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_ff0);
    ggml_tensor * up1_w      = ggml_new_tensor_2d(ctx, weight_type, n_embd, n_ff1);
    ggml_tensor * down0_w    = ggml_new_tensor_2d(ctx, weight_type, n_ff0, n_embd);
    ggml_tensor * down1_w    = ggml_new_tensor_2d(ctx, weight_type, n_ff1, n_embd);
    ggml_tensor * shexp0_w   = ggml_new_tensor_2d(ctx, weight_type, n_embd, 1);
    ggml_tensor * shexp1_w   = ggml_new_tensor_2d(ctx, weight_type, n_embd, 1);

    ggml_tensor * routed = ggml_mul_mat(ctx, routed_w, x);
    ggml_tensor * up0    = ggml_mul_mat(ctx, up0_w, x);
    ggml_tensor * gate0  = ggml_mul_mat(ctx, gate0_w, x);
    ggml_tensor * silu0  = ggml_silu(ctx, gate0);
    ggml_tensor * hid0   = ggml_mul(ctx, silu0, up0);
    ggml_tensor * down0  = ggml_mul_mat(ctx, down0_w, hid0);
    ggml_tensor * sg0    = ggml_mul_mat(ctx, shexp0_w, x);
    ggml_tensor * sig0   = ggml_sigmoid(ctx, sg0);
    ggml_tensor * gated0 = ggml_mul(ctx, down0, sig0);
    ggml_tensor * out0   = ggml_add(ctx, gated0, routed);

    ggml_tensor * up1    = ggml_mul_mat(ctx, up1_w, x);
    ggml_tensor * gate1  = ggml_mul_mat(ctx, gate1_w, x);
    ggml_tensor * silu1  = ggml_silu(ctx, gate1);
    ggml_tensor * hid1   = ggml_mul(ctx, silu1, up1);
    ggml_tensor * down1  = ggml_mul_mat(ctx, down1_w, hid1);
    ggml_tensor * sg1    = ggml_mul_mat(ctx, shexp1_w, x);
    ggml_tensor * sig1   = ggml_sigmoid(ctx, sg1);
    ggml_tensor * out1   = ggml_mul(ctx, down1, sig1);

    ggml_tensor * srcs[2] = { out0, out1 };
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
    ggml_backend_sched_set_tensor_backend(sched, routed_w, backends[2]);
    ggml_backend_sched_set_tensor_backend(sched, routed,   backends[2]);
    ggml_backend_sched_set_tensor_backend(sched, gate0_w,  backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, up0_w,    backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, down0_w,  backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, shexp0_w, backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, up0,      backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, gate0,    backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, silu0,    backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, hid0,     backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, down0,    backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, sg0,      backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, sig0,     backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, gated0,   backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, out0,     backends[0]);
    ggml_backend_sched_set_tensor_backend(sched, gate1_w,  backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, up1_w,    backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, down1_w,  backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, shexp1_w, backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, up1,      backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, gate1,    backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, silu1,    backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, hid1,     backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, down1,    backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, sg1,      backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, sig1,     backends[1]);
    ggml_backend_sched_set_tensor_backend(sched, out1,     backends[1]);
    require(ggml_backend_sched_alloc_graph(sched, gf), "failed to allocate scheduler shared-expert split graph");

    std::vector<float> xv(ggml_nelements(x));
    for (size_t i = 0; i < xv.size(); ++i) {
        xv[i] = 0.03f + 0.01f*(float) i;
    }
    std::vector<float> routedv(ggml_nelements(routed_w));
    std::vector<float> gate0v(ggml_nelements(gate0_w));
    std::vector<float> gate1v(ggml_nelements(gate1_w));
    std::vector<float> up0v(ggml_nelements(up0_w));
    std::vector<float> up1v(ggml_nelements(up1_w));
    std::vector<float> down0v(ggml_nelements(down0_w));
    std::vector<float> down1v(ggml_nelements(down1_w));
    std::vector<float> shexpv(ggml_nelements(shexp0_w));
    std::vector<float> gatev((size_t) n_embd*n_ff);
    std::vector<float> upv((size_t) n_embd*n_ff);
    std::vector<float> downv((size_t) n_ff*n_embd);
    for (size_t i = 0; i < routedv.size(); ++i) routedv[i] = 0.001f + 0.0001f*(float) i;
    for (size_t i = 0; i < gate0v.size(); ++i)  gate0v[i]  = 0.002f + 0.0002f*(float) i;
    for (size_t i = 0; i < gate1v.size(); ++i)  gate1v[i]  = 0.003f + 0.0003f*(float) i;
    for (size_t i = 0; i < up0v.size(); ++i)    up0v[i]    = 0.004f + 0.0004f*(float) i;
    for (size_t i = 0; i < up1v.size(); ++i)    up1v[i]    = 0.005f + 0.0005f*(float) i;
    for (size_t i = 0; i < down0v.size(); ++i)  down0v[i]  = 0.006f + 0.0006f*(float) i;
    for (size_t i = 0; i < down1v.size(); ++i)  down1v[i]  = 0.007f + 0.0007f*(float) i;
    for (size_t i = 0; i < shexpv.size(); ++i)  shexpv[i]  = -0.01f + 0.0005f*(float) i;

    for (int64_t ih = 0; ih < n_ff0; ++ih) {
        for (int64_t ic = 0; ic < n_embd; ++ic) {
            gatev[idx2(ic, ih, n_embd)] = gate0v[idx2(ic, ih, n_embd)];
            upv[idx2(ic, ih, n_embd)]   = up0v[idx2(ic, ih, n_embd)];
        }
    }
    for (int64_t ih = 0; ih < n_ff1; ++ih) {
        for (int64_t ic = 0; ic < n_embd; ++ic) {
            gatev[idx2(ic, n_ff0 + ih, n_embd)] = gate1v[idx2(ic, ih, n_embd)];
            upv[idx2(ic, n_ff0 + ih, n_embd)]   = up1v[idx2(ic, ih, n_embd)];
        }
    }
    for (int64_t io = 0; io < n_embd; ++io) {
        for (int64_t ih = 0; ih < n_ff0; ++ih) {
            downv[idx2(ih, io, n_ff)] = down0v[idx2(ih, io, n_ff0)];
        }
        for (int64_t ih = 0; ih < n_ff1; ++ih) {
            downv[idx2(n_ff0 + ih, io, n_ff)] = down1v[idx2(ih, io, n_ff1)];
        }
    }

    std::vector<uint8_t> routedq(ggml_nbytes(routed_w));
    std::vector<uint8_t> gate0q(ggml_nbytes(gate0_w));
    std::vector<uint8_t> gate1q(ggml_nbytes(gate1_w));
    std::vector<uint8_t> up0q(ggml_nbytes(up0_w));
    std::vector<uint8_t> up1q(ggml_nbytes(up1_w));
    std::vector<uint8_t> down0q(ggml_nbytes(down0_w));
    std::vector<uint8_t> down1q(ggml_nbytes(down1_w));
    std::vector<uint8_t> shexpq(ggml_nbytes(shexp0_w));
    ggml_quantize_chunk(weight_type, routedv.data(), routedq.data(), 0, ggml_nrows(routed_w), routed_w->ne[0], nullptr, nullptr);
    ggml_quantize_chunk(weight_type, gate0v.data(),  gate0q.data(),  0, ggml_nrows(gate0_w),  gate0_w->ne[0],  nullptr, nullptr);
    ggml_quantize_chunk(weight_type, gate1v.data(),  gate1q.data(),  0, ggml_nrows(gate1_w),  gate1_w->ne[0],  nullptr, nullptr);
    ggml_quantize_chunk(weight_type, up0v.data(),    up0q.data(),    0, ggml_nrows(up0_w),    up0_w->ne[0],    nullptr, nullptr);
    ggml_quantize_chunk(weight_type, up1v.data(),    up1q.data(),    0, ggml_nrows(up1_w),    up1_w->ne[0],    nullptr, nullptr);
    ggml_quantize_chunk(weight_type, down0v.data(),  down0q.data(),  0, ggml_nrows(down0_w),  down0_w->ne[0],  nullptr, nullptr);
    ggml_quantize_chunk(weight_type, down1v.data(),  down1q.data(),  0, ggml_nrows(down1_w),  down1_w->ne[0],  nullptr, nullptr);
    ggml_quantize_chunk(weight_type, shexpv.data(),  shexpq.data(),  0, ggml_nrows(shexp0_w), shexp0_w->ne[0], nullptr, nullptr);

    ggml_backend_tensor_set(x,        xv.data(),      0, ggml_nbytes(x));
    ggml_backend_tensor_set(routed_w, routedq.data(), 0, ggml_nbytes(routed_w));
    ggml_backend_tensor_set(gate0_w,  gate0q.data(),  0, ggml_nbytes(gate0_w));
    ggml_backend_tensor_set(gate1_w,  gate1q.data(),  0, ggml_nbytes(gate1_w));
    ggml_backend_tensor_set(up0_w,    up0q.data(),    0, ggml_nbytes(up0_w));
    ggml_backend_tensor_set(up1_w,    up1q.data(),    0, ggml_nbytes(up1_w));
    ggml_backend_tensor_set(down0_w,  down0q.data(),  0, ggml_nbytes(down0_w));
    ggml_backend_tensor_set(down1_w,  down1q.data(),  0, ggml_nbytes(down1_w));
    ggml_backend_tensor_set(shexp0_w, shexpq.data(),  0, ggml_nbytes(shexp0_w));
    ggml_backend_tensor_set(shexp1_w, shexpq.data(),  0, ggml_nbytes(shexp1_w));

    require(ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS, "scheduler shared-expert split graph failed");

    std::vector<float> got(ggml_nelements(out), 0.0f);
    ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));

    ggml_context * ctx_ref = ggml_init(params);
    require(ctx_ref != nullptr, "failed to initialize reference ggml context");
    ggml_tensor * x_ref       = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * routed_ref_w = ggml_new_tensor_2d(ctx_ref, weight_type, n_embd, n_embd);
    ggml_tensor * gate_ref_w  = ggml_new_tensor_2d(ctx_ref, weight_type, n_embd, n_ff);
    ggml_tensor * up_ref_w    = ggml_new_tensor_2d(ctx_ref, weight_type, n_embd, n_ff);
    ggml_tensor * down_ref_w  = ggml_new_tensor_2d(ctx_ref, weight_type, n_ff, n_embd);
    ggml_tensor * shexp_ref_w = ggml_new_tensor_2d(ctx_ref, weight_type, n_embd, 1);
    ggml_tensor * routed_ref  = ggml_mul_mat(ctx_ref, routed_ref_w, x_ref);
    ggml_tensor * up_ref      = ggml_mul_mat(ctx_ref, up_ref_w, x_ref);
    ggml_tensor * gate_ref    = ggml_mul_mat(ctx_ref, gate_ref_w, x_ref);
    ggml_tensor * silu_ref    = ggml_silu(ctx_ref, gate_ref);
    ggml_tensor * hid_ref     = ggml_mul(ctx_ref, silu_ref, up_ref);
    ggml_tensor * shared_ref  = ggml_mul_mat(ctx_ref, down_ref_w, hid_ref);
    ggml_tensor * sg_ref      = ggml_mul_mat(ctx_ref, shexp_ref_w, x_ref);
    ggml_tensor * sig_ref     = ggml_sigmoid(ctx_ref, sg_ref);
    shared_ref = ggml_mul(ctx_ref, shared_ref, sig_ref);
    ggml_tensor * out_ref = ggml_add(ctx_ref, routed_ref, shared_ref);
    ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(gf_ref, out_ref);

    ggml_backend_t backend_ref = ggml_backend_cpu_init();
    require(backend_ref != nullptr, "failed to initialize reference CPU backend");
    ggml_backend_buffer_t buffer_ref = ggml_backend_alloc_ctx_tensors(ctx_ref, backend_ref);
    require(buffer_ref != nullptr, "failed to allocate reference CPU tensors");

    std::vector<uint8_t> gateq(ggml_nbytes(gate_ref_w));
    std::vector<uint8_t> upq(ggml_nbytes(up_ref_w));
    std::vector<uint8_t> downq(ggml_nbytes(down_ref_w));
    ggml_quantize_chunk(weight_type, gatev.data(), gateq.data(), 0, ggml_nrows(gate_ref_w), gate_ref_w->ne[0], nullptr, nullptr);
    ggml_quantize_chunk(weight_type, upv.data(),   upq.data(),   0, ggml_nrows(up_ref_w),   up_ref_w->ne[0],   nullptr, nullptr);
    ggml_quantize_chunk(weight_type, downv.data(), downq.data(), 0, ggml_nrows(down_ref_w), down_ref_w->ne[0], nullptr, nullptr);

    ggml_backend_tensor_set(x_ref,        xv.data(),      0, ggml_nbytes(x_ref));
    ggml_backend_tensor_set(routed_ref_w, routedq.data(), 0, ggml_nbytes(routed_ref_w));
    ggml_backend_tensor_set(gate_ref_w,   gateq.data(),   0, ggml_nbytes(gate_ref_w));
    ggml_backend_tensor_set(up_ref_w,     upq.data(),     0, ggml_nbytes(up_ref_w));
    ggml_backend_tensor_set(down_ref_w,   downq.data(),   0, ggml_nbytes(down_ref_w));
    ggml_backend_tensor_set(shexp_ref_w,  shexpq.data(),  0, ggml_nbytes(shexp_ref_w));
    require(ggml_backend_graph_compute(backend_ref, gf_ref) == GGML_STATUS_SUCCESS, "reference shared-expert graph failed");

    std::vector<float> want(ggml_nelements(out_ref), 0.0f);
    ggml_backend_tensor_get(out_ref, want.data(), 0, ggml_nbytes(out_ref));

    for (size_t i = 0; i < got.size(); ++i) {
        require_near_tol(got[i], want[i], 5e-3f, "scheduler shared-expert split mismatch");
    }

    ggml_backend_buffer_free(buffer_ref);
    ggml_backend_free(backend_ref);
    ggml_free(ctx_ref);

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
    roundtrip_split_bytes(GGML_TYPE_Q8_0, 0);
    roundtrip_split_bytes(GGML_TYPE_Q8_0, 1);
    roundtrip_split_bytes(GGML_TYPE_Q8_0, 2);
    test_reduce_add_cpu_backend();
    test_scheduler_reduce_on_numa_backends();
    test_scheduler_moe_hidden_split();
    test_scheduler_shared_expert_gate_split();
    std::puts("cpu-numa-tp tests passed");
    return 0;
}
