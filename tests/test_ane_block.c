/* Block parity gate: one H3 transformer block, pure Metal against the same
 * block with its four projections on the Neural Engine.
 *
 * usage: h3_ane_block_test [ROWS]
 *
 * The released checkpoint is not on this machine, so the block runs on pinned
 * pseudo-random weights at the exact DiT core shapes. The op sequence is
 * h3_dit.c's refiner block: RMSNorm, QKV, Q/K norm, SDPA, output projection,
 * residual, RMSNorm, FC1, SwiGLU, FC2, residual. */

#include "h3_ane_linear.h"
#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336
};

static double seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static uint64_t state = 0x1234567890abcdefull;

static float next_uniform(float scale) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return ((float)(state >> 40) / 8388608.0f - 1.0f) * scale;
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static h3_gpu_tensor *random_bf16(h3_gpu *gpu, size_t elements, float scale) {
    uint16_t *values = malloc(elements * sizeof(uint16_t));
    if (!values) return NULL;
    for (size_t i = 0; i < elements; i++)
        values[i] = to_bf16(next_uniform(scale));
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, values, elements);
    free(values);
    return tensor;
}

static h3_gpu_tensor *ones_bf16(h3_gpu *gpu, size_t elements) {
    uint16_t *values = malloc(elements * sizeof(uint16_t));
    if (!values) return NULL;
    for (size_t i = 0; i < elements; i++)
        values[i] = to_bf16(1.0f + next_uniform(0.05f));
    h3_gpu_tensor *tensor = h3_gpu_tensor_from_bf16(gpu, values, elements);
    free(values);
    return tensor;
}

typedef struct {
    h3_gpu_tensor *qkv_weight;
    h3_gpu_tensor *out_weight;
    h3_gpu_tensor *fc1_weight;
    h3_gpu_tensor *fc2_weight;
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *source;
    h3_gpu_tensor *norm;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *heads;
    h3_gpu_tensor *branch;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *activated;
    h3_ane_projection *ane_qkv;
    h3_ane_projection *ane_out;
    h3_ane_projection *ane_fc1;
    h3_ane_projection *ane_fc2;
    uint32_t rows;
} block;

#define CHECK(call, label) do {                                               \
    if (!(call)) {                                                            \
        fprintf(stderr, "%s failed: %s\n", label, h3_gpu_error(gpu));         \
        return 0;                                                             \
    }                                                                         \
} while (0)

/* mode 0: the unfused BF16 sequence the splice replaces. mode 1: the four
 * projections on the Neural Engine. mode 2: h3.c's default M4 path, whose QKV
 * and MLP are fused. */
static int run_block(h3_gpu *gpu, block *b, int mode) {
    uint32_t rows = b->rows;
    int use_ane = mode == 1;
    char error[256] = {0};
    CHECK(h3_gpu_begin(gpu), "begin");
    CHECK(h3_gpu_copy_bf16(gpu, b->hidden, 0, b->source, 0,
                           (size_t)rows * HIDDEN), "seed hidden");
    CHECK(h3_gpu_rms_norm_bf16(gpu, b->norm, b->hidden, b->norm1, rows, HIDDEN,
                               1e-5f), "attention norm");
    if (mode == 2) {
        CHECK(h3_gpu_grouped_qkv_linear_rope_bf16(
                  gpu, b->query, b->key, b->value, b->qkv, b->norm,
                  b->qkv_weight, b->q_norm, b->k_norm, b->q_norm, b->q_norm,
                  rows, HIDDEN, HEADS, HEAD_DIM, 0, 1e-5f), "fused QKV");
    } else if (use_ane) {
        if (!h3_ane_projection_apply(b->ane_qkv, gpu, b->qkv, b->norm, error,
                                     sizeof(error))) {
            fprintf(stderr, "ANE QKV failed: %s\n", error);
            return 0;
        }
    } else {
        CHECK(h3_gpu_linear_bf16(gpu, b->qkv, b->norm, b->qkv_weight, NULL,
                                 rows, HIDDEN, INNER * 3), "QKV");
    }
    if (mode != 2)
        CHECK(h3_gpu_grouped_qkv_rope_bf16(gpu, b->query, b->key, b->value,
                                           b->qkv, b->q_norm, b->k_norm,
                                           b->q_norm, b->q_norm, rows, HEADS,
                                           HEAD_DIM, 0, 1e-5f), "QK norm");
    CHECK(h3_gpu_sdpa_bf16(gpu, b->heads, b->query, b->key, b->value, rows,
                           HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
          "attention");
    if (use_ane) {
        if (!h3_ane_projection_apply(b->ane_out, gpu, b->branch, b->heads,
                                     error, sizeof(error))) {
            fprintf(stderr, "ANE output failed: %s\n", error);
            return 0;
        }
    } else {
        CHECK(h3_gpu_linear_bf16(gpu, b->branch, b->heads, b->out_weight, NULL,
                                 rows, INNER, HIDDEN), "attention output");
    }
    CHECK(h3_gpu_add_bf16(gpu, b->hidden, b->hidden, b->branch,
                          rows * HIDDEN), "attention residual");
    CHECK(h3_gpu_rms_norm_bf16(gpu, b->norm, b->hidden, b->norm2, rows, HIDDEN,
                               1e-5f), "MLP norm");
    if (mode == 2) {
        CHECK(h3_gpu_mlp_bf16(gpu, b->branch, b->norm, b->fc1_weight,
                              b->fc2_weight, rows, HIDDEN, FFN, HIDDEN),
              "fused MLP");
    } else if (use_ane) {
        if (!h3_ane_projection_apply(b->ane_fc1, gpu, b->fc1, b->norm, error,
                                     sizeof(error))) {
            fprintf(stderr, "ANE FC1 failed: %s\n", error);
            return 0;
        }
    } else {
        CHECK(h3_gpu_linear_bf16(gpu, b->fc1, b->norm, b->fc1_weight, NULL,
                                 rows, HIDDEN, FFN * 2), "MLP input");
    }
    if (mode != 2)
        CHECK(h3_gpu_swiglu_bf16(gpu, b->activated, b->fc1, rows, FFN),
              "SwiGLU");
    if (mode == 2) {
        /* the fused MLP already wrote the branch */
    } else if (use_ane) {
        if (!h3_ane_projection_apply(b->ane_fc2, gpu, b->branch, b->activated,
                                     error, sizeof(error))) {
            fprintf(stderr, "ANE FC2 failed: %s\n", error);
            return 0;
        }
    } else {
        CHECK(h3_gpu_linear_bf16(gpu, b->branch, b->activated, b->fc2_weight,
                                 NULL, rows, FFN, HIDDEN), "MLP output");
    }
    CHECK(h3_gpu_add_bf16(gpu, b->hidden, b->hidden, b->branch,
                          rows * HIDDEN), "MLP residual");
    CHECK(h3_gpu_submit(gpu), "submit");
    return 1;
}

static double measure(h3_gpu *gpu, block *b, int use_ane, int repeats,
                      double *mean) {
    double best = 1e30, total = 0.0;
    for (int i = 0; i < repeats; i++) {
        double started = seconds();
        if (!run_block(gpu, b, use_ane)) return -1.0;
        double elapsed = seconds() - started;
        total += elapsed;
        if (elapsed < best) best = elapsed;
    }
    if (mean) *mean = total / (double)repeats;
    return best;
}

int main(int argc, char **argv) {
    uint32_t rows = argc > 1 ? (uint32_t)atoi(argv[1]) : 1536;
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "cannot create the Metal context: %s\n", error);
        return 2;
    }
    if (!h3_ane_linear_available()) {
        fprintf(stderr, "the Neural Engine bridge is unavailable\n");
        return 2;
    }
    block b = {0};
    b.rows = rows;
    b.qkv_weight = random_bf16(gpu, (size_t)INNER * 3 * HIDDEN, 0.02f);
    b.out_weight = random_bf16(gpu, (size_t)HIDDEN * INNER, 0.02f);
    b.fc1_weight = random_bf16(gpu, (size_t)FFN * 2 * HIDDEN, 0.02f);
    b.fc2_weight = random_bf16(gpu, (size_t)HIDDEN * FFN, 0.02f);
    b.norm1 = ones_bf16(gpu, HIDDEN);
    b.norm2 = ones_bf16(gpu, HIDDEN);
    b.q_norm = ones_bf16(gpu, HEAD_DIM);
    b.k_norm = ones_bf16(gpu, HEAD_DIM);
    b.source = random_bf16(gpu, (size_t)rows * HIDDEN, 1.0f);
    b.hidden = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * HIDDEN);
    b.norm = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * HIDDEN);
    b.qkv = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * INNER * 3);
    b.query = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * INNER);
    b.key = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * INNER);
    b.value = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * INNER);
    b.heads = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * INNER);
    b.branch = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * HIDDEN);
    b.fc1 = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * FFN * 2);
    b.activated = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * FFN);
    if (!b.qkv_weight || !b.out_weight || !b.fc1_weight || !b.fc2_weight ||
        !b.norm1 || !b.norm2 || !b.q_norm || !b.k_norm || !b.source ||
        !b.hidden || !b.norm || !b.qkv || !b.query || !b.key || !b.value ||
        !b.heads || !b.branch || !b.fc1 || !b.activated) {
        fprintf(stderr, "cannot allocate the block: %s\n", h3_gpu_error(gpu));
        return 2;
    }

    double metal_mean = 0.0, ane_mean = 0.0, fused_mean = 0.0;
    double fused_time = measure(gpu, &b, 2, 3, &fused_mean);
    if (fused_time < 0.0) return 1;
    double metal_time = measure(gpu, &b, 0, 3, &metal_mean);
    if (metal_time < 0.0) return 1;
    size_t elements = (size_t)rows * HIDDEN;
    float *reference = malloc(elements * sizeof(float));
    uint16_t *raw = malloc(elements * sizeof(uint16_t));
    if (!reference || !raw) return 2;
    if (!h3_gpu_tensor_read_bf16(b.hidden, raw, elements)) return 1;
    for (size_t i = 0; i < elements; i++) reference[i] = from_bf16(raw[i]);

    double compile_started = seconds();
    b.ane_qkv = h3_ane_projection_create(gpu, "qkv", b.qkv_weight, HIDDEN,
        INNER * 3, rows, h3_ane_linear_default_chunk(HIDDEN), error,
        sizeof(error));
    b.ane_out = b.ane_qkv ? h3_ane_projection_create(gpu, "out", b.out_weight,
        INNER, HIDDEN, rows, h3_ane_linear_default_chunk(INNER), error,
        sizeof(error)) : NULL;
    b.ane_fc1 = b.ane_out ? h3_ane_projection_create(gpu, "fc1", b.fc1_weight,
        HIDDEN, FFN * 2, rows, h3_ane_linear_default_chunk(HIDDEN), error,
        sizeof(error)) : NULL;
    b.ane_fc2 = b.ane_fc1 ? h3_ane_projection_create(gpu, "fc2", b.fc2_weight,
        FFN, HIDDEN, rows, h3_ane_linear_default_chunk(FFN), error,
        sizeof(error)) : NULL;
    if (!b.ane_fc2) {
        fprintf(stderr, "cannot build the ANE projections: %s\n", error);
        return 1;
    }
    double compile_time = seconds() - compile_started;
    printf("rows=%u  chunks qkv=%u out=%u fc1=%u fc2=%u  "
           "ANE weights=%.1f MiB  build=%.2f s\n", rows,
           h3_ane_linear_default_chunk(HIDDEN) ?
               (HIDDEN + h3_ane_linear_default_chunk(HIDDEN) - 1) /
               h3_ane_linear_default_chunk(HIDDEN) : 0,
           (INNER + h3_ane_linear_default_chunk(INNER) - 1) /
               h3_ane_linear_default_chunk(INNER),
           (HIDDEN + h3_ane_linear_default_chunk(HIDDEN) - 1) /
               h3_ane_linear_default_chunk(HIDDEN),
           (FFN + h3_ane_linear_default_chunk(FFN) - 1) /
               h3_ane_linear_default_chunk(FFN),
           (double)(h3_ane_projection_weight_bytes(b.ane_qkv) +
                    h3_ane_projection_weight_bytes(b.ane_out) +
                    h3_ane_projection_weight_bytes(b.ane_fc1) +
                    h3_ane_projection_weight_bytes(b.ane_fc2)) /
               (1024.0 * 1024.0), compile_time);

    double ane_time = measure(gpu, &b, 1, 3, &ane_mean);
    if (ane_time < 0.0) return 1;
    if (!h3_gpu_tensor_read_bf16(b.hidden, raw, elements)) return 1;

    double dot = 0.0, left = 0.0, right = 0.0, worst = 0.0, error_sum = 0.0;
    size_t nonfinite = 0;
    for (size_t i = 0; i < elements; i++) {
        double a = from_bf16(raw[i]), r = reference[i];
        if (!isfinite(a)) { nonfinite++; continue; }
        dot += a * r;
        left += a * a;
        right += r * r;
        double difference = fabs(a - r);
        error_sum += difference * difference;
        if (difference > worst) worst = difference;
    }
    double cosine = dot / (sqrt(left) * sqrt(right) + 1e-30);
    double relative = sqrt(error_sum / (right + 1e-30));
    int pass = cosine >= 0.999 && nonfinite == 0;
    printf("block parity cos=%.6f rel_l2=%.3e max_abs=%.3e nonfinite=%zu %s\n",
           cosine, relative, worst, nonfinite, pass ? "PASS" : "FAIL");
    printf("block wall  metal_default=%.1f ms  metal_unfused=%.1f ms  "
           "ane=%.1f ms  ratio_vs_default=%.2fx\n",
           fused_time * 1e3, metal_time * 1e3, ane_time * 1e3,
           fused_time / ane_time);
    printf("block mean  metal_default=%.1f ms  metal_unfused=%.1f ms  "
           "ane=%.1f ms\n", fused_mean * 1e3, metal_mean * 1e3,
           ane_mean * 1e3);
    const char *names[4] = {"qkv", "out", "fc1", "fc2"};
    h3_ane_projection *all[4] = {b.ane_qkv, b.ane_out, b.ane_fc1, b.ane_fc2};
    double pack_total = 0.0, eval_total = 0.0;
    for (int i = 0; i < 4; i++) {
        double pack = 0.0, eval = 0.0;
        uint64_t calls = 0;
        if (getenv("H3_ANE_PROFILE_STAGES")) {
            double sync = 0.0, unpack = 0.0;
            h3_ane_projection_stage_timings(
                all[i], &sync, &pack, &eval, &unpack, &calls);
            if (calls)
                printf("  %-3s sync=%.2f ms pack=%.2f ms ane=%.2f ms "
                       "unpack=%.2f ms\n", names[i],
                       sync / (double)calls * 1e3,
                       pack / (double)calls * 1e3,
                       eval / (double)calls * 1e3,
                       unpack / (double)calls * 1e3);
        } else {
            h3_ane_projection_timings(all[i], &pack, &eval, &calls);
        }
        if (!calls) continue;
        pack_total += pack / (double)calls;
        eval_total += eval / (double)calls;
        printf("  %-3s pack+submit=%.2f ms  ane=%.2f ms  compile=%.2f s\n",
               names[i], pack / (double)calls * 1e3,
               eval / (double)calls * 1e3,
               h3_ane_projection_compile_seconds(all[i]));
    }
    printf("  mean run: pack+submit=%.1f ms  neural engine=%.1f ms  "
           "rest=%.1f ms\n", pack_total * 1e3, eval_total * 1e3,
           (ane_mean - pack_total - eval_total) * 1e3);
    h3_ane_projection_free(b.ane_qkv);
    h3_ane_projection_free(b.ane_out);
    h3_ane_projection_free(b.ane_fc1);
    h3_ane_projection_free(b.ane_fc2);
    free(reference);
    free(raw);
    return pass ? 0 : 1;
}
