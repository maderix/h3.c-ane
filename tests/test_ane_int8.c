/* INT8 ConvRot gates for the Neural Engine projection module.
 *
 * usage: h3_ane_int8_test [CHECKPOINT_DIR]
 *
 * Synthetic gates mint int8_tensorwise payloads with the reference formula in
 * double precision and compare the ANE result against a double replay of the
 * stored contract (fp16-rounded scale, dequantize, derotate, matmul). With a
 * checkpoint directory, the full block-0 qkv projection runs on real weights. */

#include "h3_ane_linear.h"
#include "h3_convrot.h"
#include "h3_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned failures;

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static double rng_uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return ((double)((rng_state * 2685821657736338717ull) >> 11) /
            9007199254740992.0) - 0.5;
}

/* Double replay of the stored contract for one projection. */
static double *reference(const int8_t *quantized, const float *scales,
                         int group_size, uint32_t input_dim,
                         uint32_t output_dim, const float *x, uint32_t rows) {
    const float *table = group_size ? h3_convrot_hadamard(group_size) : NULL;
    double *weights = malloc(sizeof(double) * output_dim * input_dim);
    double *result = malloc(sizeof(double) * output_dim * rows);
    if (!weights || !result) { free(weights); free(result); return NULL; }
    for (uint32_t n = 0; n < output_dim; n++) {
        double scale = (double)(__fp16)scales[n];
        for (uint32_t k = 0; k < input_dim; k++) {
            if (!group_size) {
                weights[(size_t)n * input_dim + k] =
                    (double)quantized[(size_t)n * input_dim + k] * scale;
                continue;
            }
            uint32_t group = k / (uint32_t)group_size;
            uint32_t j = k % (uint32_t)group_size;
            double acc = 0;
            for (int i = 0; i < group_size; i++)
                acc += (double)quantized[(size_t)n * input_dim +
                                         group * group_size + i] * scale *
                       (double)table[i * group_size + (int)j];
            weights[(size_t)n * input_dim + k] = acc;
        }
    }
    for (uint32_t n = 0; n < output_dim; n++)
        for (uint32_t s = 0; s < rows; s++) {
            double acc = 0;
            for (uint32_t k = 0; k < input_dim; k++)
                acc += weights[(size_t)n * input_dim + k] *
                       (double)x[(size_t)k * rows + s];
            result[(size_t)n * rows + s] = acc;
        }
    free(weights);
    return result;
}

static void gate(const char *name, const int8_t *quantized,
                 const float *scales, int group_size, uint32_t input_dim,
                 uint32_t output_dim, uint32_t rows, uint32_t kc) {
    char error[256] = {0};
    h3_ane_linear *linear = h3_ane_linear_create_int8(
        name, quantized, scales, group_size, input_dim, output_dim, rows, kc,
        error, sizeof(error));
    if (!linear) {
        printf("FAIL %s create: %s\n", name, error);
        failures++;
        return;
    }
    uint32_t chunks = h3_ane_linear_chunks(linear);
    uint32_t chunk_dim = h3_ane_linear_chunk_dim(linear);
    uint32_t plane_rows = h3_ane_linear_plane_rows(linear);
    float *x = malloc(sizeof(float) * input_dim * rows);
    for (size_t i = 0; i < (size_t)input_dim * rows; i++)
        x[i] = (float)(rng_uniform() * 4.0);
    for (uint32_t c = 0; c < chunks; c++) {
        float *plane = h3_ane_linear_input(linear, c);
        memset(plane, 0, h3_ane_linear_input_bytes(linear));
        uint32_t base = c * chunk_dim;
        uint32_t span = input_dim > base ?
            (input_dim - base < chunk_dim ? input_dim - base : chunk_dim) : 0;
        for (uint32_t k = 0; k < span; k++)
            memcpy(plane + (size_t)k * plane_rows,
                   x + (size_t)(base + k) * rows, sizeof(float) * rows);
    }
    if (!h3_ane_linear_eval(linear, error, sizeof(error))) {
        printf("FAIL %s eval: %s\n", name, error);
        failures++;
        h3_ane_linear_free(linear);
        free(x);
        return;
    }
    double *want = reference(quantized, scales, group_size, input_dim,
                             output_dim, x, rows);
    const float *got = h3_ane_linear_output(linear);
    double dot = 0, got_norm = 0, want_norm = 0, error_sum = 0, scale_sum = 0;
    double worst = 0;
    size_t nonfinite = 0;
    for (uint32_t n = 0; n < output_dim; n++)
        for (uint32_t s = 0; s < rows; s++) {
            double a = got[(size_t)n * plane_rows + s];
            double b = want[(size_t)n * rows + s];
            if (!isfinite(a)) { nonfinite++; continue; }
            dot += a * b;
            got_norm += a * a;
            want_norm += b * b;
            double difference = a - b;
            error_sum += difference * difference;
            scale_sum += b * b;
            if (fabs(difference) > worst) worst = fabs(difference);
        }
    double cosine = dot / (sqrt(got_norm) * sqrt(want_norm) + 1e-30);
    double relative = sqrt(error_sum / (scale_sum + 1e-30));
    int pass = cosine >= 0.9999 && relative <= 2e-3 && nonfinite == 0;
    printf("%-12s K=%u N=%u kc=%u rows=%u gs=%d: cos=%.7f rel_l2=%.3e "
           "max_abs=%.3e nonfinite=%zu weights=%.1f MiB %s\n",
           name, input_dim, output_dim, kc, rows, group_size, cosine, relative,
           worst, nonfinite,
           (double)h3_ane_linear_weight_bytes(linear) / (1024.0 * 1024.0),
           pass ? "PASS" : "FAIL");
    if (!pass) failures++;
    h3_ane_linear_free(linear);
    free(x);
    free(want);
}

static void synthetic(const char *name, uint32_t input_dim,
                      uint32_t output_dim, uint32_t rows, uint32_t kc,
                      int group_size) {
    const float *table = group_size ? h3_convrot_hadamard(group_size) : NULL;
    double *original = malloc(sizeof(double) * output_dim * input_dim);
    double *stored = malloc(sizeof(double) * output_dim * input_dim);
    int8_t *quantized = malloc((size_t)output_dim * input_dim);
    float *scales = malloc(sizeof(float) * output_dim);
    for (size_t i = 0; i < (size_t)output_dim * input_dim; i++)
        original[i] = rng_uniform();
    for (uint32_t n = 0; n < output_dim; n++)
        for (uint32_t k = 0; k < input_dim; k++) {
            if (!group_size) {
                stored[(size_t)n * input_dim + k] =
                    original[(size_t)n * input_dim + k];
                continue;
            }
            uint32_t group = k / (uint32_t)group_size;
            uint32_t i = k % (uint32_t)group_size;
            double acc = 0;
            for (int j = 0; j < group_size; j++)
                acc += original[(size_t)n * input_dim + group * group_size + j] *
                       (double)table[(int)i * group_size + j];
            stored[(size_t)n * input_dim + k] = acc;
        }
    for (uint32_t n = 0; n < output_dim; n++) {
        double amax = 0;
        for (uint32_t k = 0; k < input_dim; k++)
            amax = fmax(amax, fabs(stored[(size_t)n * input_dim + k]));
        scales[n] = (float)(amax / 127.0);
        for (uint32_t k = 0; k < input_dim; k++) {
            double q = round(stored[(size_t)n * input_dim + k] /
                             (double)scales[n]);
            quantized[(size_t)n * input_dim + k] =
                (int8_t)fmax(-128.0, fmin(127.0, q));
        }
    }
    gate(name, quantized, scales, group_size, input_dim, output_dim, rows, kc);
    free(original);
    free(stored);
    free(quantized);
    free(scales);
}

int main(int argc, char **argv) {
    if (!h3_ane_linear_available()) {
        printf("skip: the Neural Engine bridge is unavailable\n");
        return 0;
    }
    /* Two structurally different synthetic cases: a padded multi-chunk
     * rotated projection and an unrotated kc=2048 projection. */
    synthetic("rot-padded", 1280, 384, 32, 1024, 256);
    synthetic("plain-2048", 4096, 256, 64, 2048, 0);
    if (argc > 1) {
        char error[512] = {0};
        h3_weight_store *store = h3_weight_store_open(argv[1], error,
                                                      sizeof(error));
        if (!store) {
            printf("FAIL store: %s\n", error);
            return 1;
        }
        int8_t *quantized = NULL;
        float *scales = NULL;
        int group_size = 0;
        if (!h3_weight_load_int8_raw(store, "blocks.0.attn.qkv_proj.weight",
                                     21504, 5376, &quantized, &scales,
                                     &group_size, error, sizeof(error))) {
            printf("FAIL raw: %s\n", error);
            return 1;
        }
        gate("real-qkv", quantized, scales, group_size, 5376, 21504, 32,
             h3_ane_linear_default_chunk(5376));
        free(quantized);
        free(scales);
        h3_weight_store_free(store);
    }
    printf("%u failure(s)\n", failures);
    return failures ? 1 : 0;
}
