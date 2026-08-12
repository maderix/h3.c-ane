/* Full-DiT-block ANE graph gate: numerics against an f64 replay of the same
 * stored contract on real block-0 weights, then wall time at a real row count.
 *
 * usage: h3_ane_block_test CHECKPOINT_DIR [TIMING_ROWS]
 */

#include "h3_ane_block.h"
#include "h3_convrot.h"
#include "h3_weights.h"

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
    FFN = 14336,
    ROPE_HALF = 48,
    GS = 256,
    S = 64
};

static const uint32_t ENDS[3] = {6, 20, S};

static double now_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0x2545F4914F6CDD1Dull;
static double rng_uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return ((double)((rng_state * 2685821657736338717ull) >> 11) /
            9007199254740992.0) - 0.5;
}

static double f16r(double value) { return (double)(__fp16)value; }

/* Dequantized + derotated f64 weights straight from the store. */
static double *load_weights(const h3_weight_store *store, const char *name,
                            uint32_t rows, uint32_t k) {
    int8_t *quantized = NULL;
    float *scales = NULL;
    int group_size = 0;
    char error[512];
    if (!h3_weight_load_int8_raw(store, name, rows, k, &quantized, &scales,
                                 &group_size, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return NULL;
    }
    const float *table = h3_convrot_hadamard(GS);
    double *weights = malloc(sizeof(double) * rows * k);
    for (uint32_t row = 0; row < rows; row++) {
        double scale = f16r(scales[row]);   /* the graph stores fp16 scales */
        for (uint32_t group = 0; group < k / GS; group++)
            for (uint32_t j = 0; j < GS; j++) {
                double acc = 0;
                for (uint32_t i = 0; i < GS; i++)
                    acc += (double)quantized[(size_t)row * k + group * GS + i] *
                           scale * (double)table[i * GS + j];
                weights[(size_t)row * k + group * GS + j] = acc;
            }
    }
    free(quantized);
    free(scales);
    return weights;
}

static double *load_norm(const h3_weight_store *store, const char *name,
                         uint32_t width) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) return NULL;
    uint16_t *raw = malloc(width * 2);
    char error[384];
    if (!h3_st_read_data(header, tensor, raw, width * 2, error, sizeof(error)))
        return NULL;
    double *values = malloc(sizeof(double) * width);
    for (uint32_t index = 0; index < width; index++) {
        uint32_t bits = (uint32_t)raw[index] << 16;
        float value;
        memcpy(&value, &bits, sizeof(value));
        values[index] = f16r(value);        /* the graph stores fp16 gammas */
    }
    free(raw);
    return values;
}

static void rms_norm_rows(double *destination, const double *source,
                          const double *gamma, uint32_t rows, uint32_t width) {
    for (uint32_t row = 0; row < rows; row++) {
        double sum = 0;
        for (uint32_t index = 0; index < width; index++) {
            double value = source[(size_t)row * width + index];
            sum += value * value;
        }
        double inverse = 1.0 / sqrt(sum / width + 1e-5);
        for (uint32_t index = 0; index < width; index++)
            destination[(size_t)row * width + index] =
                source[(size_t)row * width + index] * inverse * gamma[index];
    }
}

static int segment_of(uint32_t row) {
    for (int seg = 0; seg < 3; seg++) if (row < ENDS[seg]) return seg;
    return 2;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s CHECKPOINT_DIR [TIMING_ROWS]\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    h3_weight_store *store = h3_weight_store_open(argv[1], error,
                                                  sizeof(error));
    if (!store) { fprintf(stderr, "%s\n", error); return 2; }

    /* Shared inputs: activations row-major [S][HIDDEN] for the reference,
     * mods [3][6][HIDDEN], fp16-rounded rope tables [S][48]. */
    static double x[S][HIDDEN];
    static double mods[3][6][HIDDEN];
    static double rope_cos[S][ROPE_HALF], rope_sin[S][ROPE_HALF];
    static float rope_cos_f[S * ROPE_HALF], rope_sin_f[S * ROPE_HALF];
    for (uint32_t row = 0; row < S; row++)
        for (uint32_t index = 0; index < HIDDEN; index++)
            x[row][index] = rng_uniform() * 2.0;
    for (int seg = 0; seg < 3; seg++)
        for (int slot = 0; slot < 6; slot++)
            for (uint32_t index = 0; index < HIDDEN; index++)
                mods[seg][slot][index] = f16r(rng_uniform());
    for (uint32_t row = 0; row < S; row++)
        for (uint32_t index = 0; index < ROPE_HALF; index++) {
            double angle = rng_uniform() * 6.0;
            rope_cos[row][index] = f16r(cos(angle));
            rope_sin[row][index] = f16r(sin(angle));
            rope_cos_f[row * ROPE_HALF + index] = (float)cos(angle);
            rope_sin_f[row * ROPE_HALF + index] = (float)sin(angle);
        }

    int staged = getenv("H3_ANE_BLOCK_STAGE") != NULL;
    /* f64 reference forward. */
    if (!staged) printf("building the f64 reference...\n");
    double *w_qkv = load_weights(store, "blocks.0.attn.qkv_proj.weight",
                                 INNER * 3, HIDDEN);
    double *w_out = load_weights(store, "blocks.0.attn.out_proj.weight",
                                 HIDDEN, INNER);
    double *w_fc1 = load_weights(store, "blocks.0.mlp.fc1.weight",
                                 FFN * 2, HIDDEN);
    double *w_fc2 = load_weights(store, "blocks.0.mlp.fc2.weight",
                                 HIDDEN, FFN);
    double *g_norm1 = load_norm(store, "blocks.0.norm1.weight", HIDDEN);
    double *g_norm2 = load_norm(store, "blocks.0.norm2.weight", HIDDEN);
    double *g_qn = load_norm(store, "blocks.0.attn.q_norm.weight", HEAD_DIM);
    double *g_kn = load_norm(store, "blocks.0.attn.k_norm.weight", HEAD_DIM);
    if (!w_qkv || !w_out || !w_fc1 || !w_fc2 || !g_norm1 || !g_norm2 ||
        !g_qn || !g_kn) return 2;

    static double normed[S][HIDDEN], modded[S][HIDDEN];
    static double n1_ref[S][HIDDEN], xm1_ref[S][HIDDEN], aout_ref[S][HIDDEN];
    static double mout_ref[S][HIDDEN];
    static double qrope_ref[S][HEADS][HEAD_DIM];
    static double act_ref[S][FFN];
    rms_norm_rows(&normed[0][0], &x[0][0], g_norm1, S, HIDDEN);
    memcpy(n1_ref, normed, sizeof(n1_ref));
    for (uint32_t row = 0; row < S; row++) {
        int seg = segment_of(row);
        for (uint32_t index = 0; index < HIDDEN; index++)
            modded[row][index] = normed[row][index] *
                (1.0 + mods[seg][1][index]) + mods[seg][0][index];
    }
    memcpy(xm1_ref, modded, sizeof(xm1_ref));
    static double qkv[S][INNER * 3];
    for (uint32_t row = 0; row < S; row++)
        for (uint32_t output = 0; output < INNER * 3; output++) {
            double acc = 0;
            for (uint32_t index = 0; index < HIDDEN; index++)
                acc += modded[row][index] * w_qkv[(size_t)output * HIDDEN + index];
            qkv[row][output] = acc;
        }
    free(w_qkv);
    /* conventional [q|k|v]; per-head norm + gamma + 48-pair rope */
    static double q[S][HEADS][HEAD_DIM], k[S][HEADS][HEAD_DIM],
                  v[S][HEADS][HEAD_DIM];
    for (uint32_t row = 0; row < S; row++)
        for (uint32_t head = 0; head < HEADS; head++)
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                q[row][head][d] = qkv[row][head * HEAD_DIM + d];
                k[row][head][d] = qkv[row][INNER + head * HEAD_DIM + d];
                v[row][head][d] = qkv[row][2 * INNER + head * HEAD_DIM + d];
            }
    for (uint32_t row = 0; row < S; row++)
        for (uint32_t head = 0; head < HEADS; head++) {
            double qs = 0, ks = 0;
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                qs += q[row][head][d] * q[row][head][d];
                ks += k[row][head][d] * k[row][head][d];
            }
            double qi = 1.0 / sqrt(qs / HEAD_DIM + 1e-5);
            double ki = 1.0 / sqrt(ks / HEAD_DIM + 1e-5);
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                q[row][head][d] *= qi * g_qn[d];
                k[row][head][d] *= ki * g_kn[d];
            }
            for (uint32_t d = 0; d < ROPE_HALF; d++) {
                double c = rope_cos[row][d], s = rope_sin[row][d];
                double q0 = q[row][head][d], q1 = q[row][head][d + ROPE_HALF];
                double k0 = k[row][head][d], k1 = k[row][head][d + ROPE_HALF];
                q[row][head][d] = q0 * c - q1 * s;
                q[row][head][d + ROPE_HALF] = q1 * c + q0 * s;
                k[row][head][d] = k0 * c - k1 * s;
                k[row][head][d + ROPE_HALF] = k1 * c + k0 * s;
            }
        }
    memcpy(qrope_ref, q, sizeof(qrope_ref));
    static double attention[S][HEADS][HEAD_DIM];
    for (uint32_t head = 0; head < HEADS; head++)
        for (uint32_t row = 0; row < S; row++) {
            double scores[S];
            double peak = -1e300;
            for (uint32_t other = 0; other < S; other++) {
                double acc = 0;
                for (uint32_t d = 0; d < HEAD_DIM; d++)
                    acc += q[row][head][d] * k[other][head][d];
                scores[other] = acc / sqrt((double)HEAD_DIM);
                if (scores[other] > peak) peak = scores[other];
            }
            double total = 0;
            for (uint32_t other = 0; other < S; other++) {
                scores[other] = exp(scores[other] - peak);
                total += scores[other];
            }
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                double acc = 0;
                for (uint32_t other = 0; other < S; other++)
                    acc += scores[other] / total * v[other][head][d];
                attention[row][head][d] = acc;
            }
        }
    static double h1[S][HIDDEN];
    for (uint32_t row = 0; row < S; row++) {
        int seg = segment_of(row);
        for (uint32_t output = 0; output < HIDDEN; output++) {
            double acc = 0;
            for (uint32_t head = 0; head < HEADS; head++)
                for (uint32_t d = 0; d < HEAD_DIM; d++)
                    acc += attention[row][head][d] *
                           w_out[(size_t)output * INNER + head * HEAD_DIM + d];
            aout_ref[row][output] = acc;
            h1[row][output] = x[row][output] + acc * mods[seg][2][output];
        }
    }
    free(w_out);
    rms_norm_rows(&normed[0][0], &h1[0][0], g_norm2, S, HIDDEN);
    for (uint32_t row = 0; row < S; row++) {
        int seg = segment_of(row);
        for (uint32_t index = 0; index < HIDDEN; index++)
            modded[row][index] = normed[row][index] *
                (1.0 + mods[seg][4][index]) + mods[seg][3][index];
    }
    static double reference[S][HIDDEN];
    for (uint32_t row = 0; row < S; row++) {
        int seg = segment_of(row);
        static double fused[FFN * 2];
        for (uint32_t output = 0; output < FFN * 2; output++) {
            double acc = 0;
            for (uint32_t index = 0; index < HIDDEN; index++)
                acc += modded[row][index] * w_fc1[(size_t)output * HIDDEN + index];
            fused[output] = acc;
        }
        double *act = act_ref[row];
        for (uint32_t index = 0; index < FFN; index++) {
            double gate = fused[index];
            act[index] = gate / (1.0 + exp(-gate)) * fused[FFN + index];
        }
        for (uint32_t output = 0; output < HIDDEN; output++) {
            double acc = 0;
            for (uint32_t index = 0; index < FFN; index++)
                acc += act[index] * w_fc2[(size_t)output * FFN + index];
            mout_ref[row][output] = acc;
            reference[row][output] = h1[row][output] +
                                     acc * mods[seg][5][output];
        }
    }
    free(w_fc1);
    free(w_fc2);
    printf("reference ready\n");

    /* ANE block eval on identical inputs. */
    h3_ane_block *block = h3_ane_block_create(
        store, "blocks.0.", S, ENDS, rope_cos_f, rope_sin_f, error,
        sizeof(error));
    if (!block) { fprintf(stderr, "create: %s\n", error); return 1; }
    printf("compiled in %.2fs, blob %.1f MiB\n",
           h3_ane_block_compile_seconds(block),
           (double)h3_ane_block_weight_bytes(block) / (1024.0 * 1024.0));
    float *input = h3_ane_block_input(block);
    float *mod_rows = h3_ane_block_mod(block);
    uint32_t padded = h3_ane_block_padded_rows(block);
    for (uint32_t index = 0; index < HIDDEN; index++)
        for (uint32_t row = 0; row < S; row++)
            input[(size_t)index * padded + row] = (float)x[row][index];
    for (int seg = 0; seg < 3; seg++)
        for (int slot = 0; slot < 6; slot++)
            for (uint32_t index = 0; index < HIDDEN; index++)
                mod_rows[((size_t)index * H3_ANE_BLOCK_MOD_WIDTH) +
                         seg * 6 + slot] = (float)mods[seg][slot][index];
    if (!h3_ane_block_eval(block, error, sizeof(error))) {
        fprintf(stderr, "eval: %s\n", error);
        return 1;
    }
    const float *output = h3_ane_block_output(block);
    if (staged) {
        int stage = atoi(getenv("H3_ANE_BLOCK_STAGE"));
        const double *want = NULL;
        uint32_t channels = HIDDEN;
        int head_layout = 0;
        switch (stage) {
        case 1: want = &n1_ref[0][0]; break;
        case 2: want = &xm1_ref[0][0]; break;
        case 3: want = &qkv[0][0]; channels = INNER * 3; break;
        case 5: want = &qrope_ref[0][0][0]; head_layout = 1; break;
        case 6: want = &attention[0][0][0]; channels = INNER; head_layout = 2; break;
        case 7: want = &aout_ref[0][0]; break;
        case 8: want = &modded[0][0]; break;
        case 9: want = &act_ref[0][0]; channels = FFN; break;
        case 10: want = &mout_ref[0][0]; break;
        default: printf("stage %d: no reference\n", stage); return 0;
        }
        double sdot = 0, sg = 0, sw = 0, sworst = 0;
        for (uint32_t row = 0; row < S; row++)
            for (uint32_t channel = 0; channel < channels; channel++) {
                double got, expect;
                if (head_layout == 1) {
                    /* [1,H,S,D] */
                    uint32_t head = channel / HEAD_DIM, d = channel % HEAD_DIM;
                    got = output[((size_t)head * S + row) * HEAD_DIM + d];
                    expect = qrope_ref[row][head][d];
                } else if (head_layout == 2) {
                    /* att_f channels = head*dim */
                    got = output[(size_t)channel * padded + row];
                    expect = attention[row][channel / HEAD_DIM][channel % HEAD_DIM];
                } else {
                    got = output[(size_t)channel * padded + row];
                    expect = want[(size_t)row * channels + channel];
                }
                sdot += got * expect; sg += got * got; sw += expect * expect;
                if (fabs(got - expect) > sworst) sworst = fabs(got - expect);
            }
        size_t infs = 0;
        double got_peak = 0, want_peak = 0;
        for (uint32_t row = 0; row < S; row++)
            for (uint32_t channel = 0; channel < channels; channel++) {
                double got = output[(size_t)channel * padded + row];
                if (!isfinite(got)) { infs++; continue; }
                if (fabs(got) > got_peak) got_peak = fabs(got);
            }
        for (uint32_t row = 0; row < S; row++)
            for (uint32_t channel = 0; channel < channels; channel++) {
                double expect = want ? want[(size_t)row * channels + channel] : 0;
                if (fabs(expect) > want_peak) want_peak = fabs(expect);
            }
        printf("stage %d: cos=%.6f max_abs=%.3e infs=%zu got_peak=%.3e "
               "want_peak=%.3e\n", stage,
               sdot / (sqrt(sg) * sqrt(sw) + 1e-30), sworst, infs, got_peak,
               want_peak);
        for (int i = 0; i < 6; i++)
            printf("  [0][%d] got=%.4f want=%.4f\n", i,
                   output[(size_t)i * padded], want[(size_t)i * channels]);
        h3_ane_block_free(block);
        h3_weight_store_free(store);
        return 0;
    }
    double dot = 0, got_norm = 0, want_norm = 0, err = 0, worst = 0;
    size_t nonfinite = 0;
    for (uint32_t row = 0; row < S; row++)
        for (uint32_t index = 0; index < HIDDEN; index++) {
            double got = output[(size_t)index * padded + row];
            double want = reference[row][index];
            if (!isfinite(got)) { nonfinite++; continue; }
            dot += got * want;
            got_norm += got * got;
            want_norm += want * want;
            double difference = got - want;
            err += difference * difference;
            if (fabs(difference) > worst) worst = fabs(difference);
        }
    for (int seg = 0; seg < 3; seg++) {
        uint32_t seg_begin = seg ? ENDS[seg - 1] : 0;
        double sdot = 0, sg = 0, sw = 0, sworst = 0;
        for (uint32_t row = seg_begin; row < ENDS[seg]; row++)
            for (uint32_t index = 0; index < HIDDEN; index++) {
                double got = output[(size_t)index * padded + row];
                double want = reference[row][index];
                sdot += got * want; sg += got * got; sw += want * want;
                if (fabs(got - want) > sworst) sworst = fabs(got - want);
            }
        printf("  segment %d rows [%u,%u): cos=%.5f max_abs=%.3e\n", seg,
               seg_begin, ENDS[seg], sdot / (sqrt(sg) * sqrt(sw) + 1e-30),
               sworst);
    }
    double cosine = dot / (sqrt(got_norm) * sqrt(want_norm) + 1e-30);
    double relative = sqrt(err / (want_norm + 1e-30));
    int pass = cosine >= 0.999 && relative <= 3e-2 && nonfinite == 0;
    printf("block S=%d: cos=%.6f rel_l2=%.3e max_abs=%.3e nonfinite=%zu %s\n",
           S, cosine, relative, worst, nonfinite, pass ? "PASS" : "FAIL");
    h3_ane_block_free(block);

    /* Timing at a production row count. */
    uint32_t timing_rows = argc > 2 ? (uint32_t)atoi(argv[2]) : 1904;
    static float zero_rope[8192 * ROPE_HALF];
    uint32_t timing_ends[3] = {6, 80, timing_rows};
    block = h3_ane_block_create(store, "blocks.0.", timing_rows, timing_ends,
                                zero_rope, zero_rope, error, sizeof(error));
    if (!block) { fprintf(stderr, "timing create: %s\n", error); return 1; }
    double best = 1e30;
    for (int repeat = 0; repeat < 3; repeat++) {
        double started = now_seconds();
        if (!h3_ane_block_eval(block, error, sizeof(error))) {
            fprintf(stderr, "timing eval: %s\n", error);
            return 1;
        }
        double elapsed = now_seconds() - started;
        if (elapsed < best) best = elapsed;
    }
    printf("timing S=%u: best %.1f ms/block (compile %.2fs)\n", timing_rows,
           best * 1e3, h3_ane_block_compile_seconds(block));

    /* Residency rotation: unload -> reload -> eval must stay bit-identical,
     * and the cycle must be cheap enough to hide behind a block eval. */
    size_t out_bytes = (size_t)HIDDEN *
        h3_ane_block_padded_rows(block) * sizeof(float);
    float *pinned = malloc(out_bytes);
    memcpy(pinned, h3_ane_block_output(block), out_bytes);
    int rotation_ok = 1;
    double unload_best = 1e30, reload_best = 1e30;
    for (int cycle = 0; cycle < 3; cycle++) {
        double started = now_seconds();
        if (!h3_ane_block_unload(block, error, sizeof(error))) {
            fprintf(stderr, "unload: %s\n", error);
            rotation_ok = 0;
            break;
        }
        double unloaded = now_seconds();
        if (!h3_ane_block_reload(block, error, sizeof(error))) {
            fprintf(stderr, "reload: %s\n", error);
            rotation_ok = 0;
            break;
        }
        double reloaded = now_seconds();
        if (!h3_ane_block_eval(block, error, sizeof(error))) {
            fprintf(stderr, "rotated eval: %s\n", error);
            rotation_ok = 0;
            break;
        }
        if (memcmp(pinned, h3_ane_block_output(block), out_bytes)) {
            fprintf(stderr, "rotated eval output differs\n");
            rotation_ok = 0;
            break;
        }
        if (unloaded - started < unload_best) unload_best = unloaded - started;
        if (reloaded - unloaded < reload_best) reload_best = reloaded - unloaded;
    }
    if (rotation_ok)
        printf("rotation S=%u: unload %.0f ms + reload %.0f ms, "
               "3 cycles bit-identical PASS\n",
               timing_rows, unload_best * 1e3, reload_best * 1e3);
    free(pinned);
    h3_ane_block_free(block);
    h3_weight_store_free(store);
    return pass && rotation_ok ? 0 : 1;
}
