/* Fixture gate for the Neural Engine projection module.
 *
 * usage: h3_ane_linear_test NAME K N KC S WEIGHTS X YREF
 *
 * WEIGHTS is ceil(K/KC) tiles of [N][KC] fp16 with the K padding zeroed, X is
 * the same number of [KC][S] F32 planes and YREF is the [N][S] F32 reference.
 * Both weight entry points are checked: the pre-chunked payload and the
 * row-major [N][K] payload that the DiT splice uses. */

#include "h3_ane_linear.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static void *read_file(const char *path, size_t expected) {
    FILE *file = fopen(path, "rb");
    if (!file) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (expected && (size_t)size != expected) {
        fprintf(stderr, "%s is %ld bytes, expected %zu\n", path, size, expected);
        fclose(file);
        return NULL;
    }
    void *data = malloc((size_t)size);
    if (data && fread(data, 1, (size_t)size, file) != (size_t)size) {
        free(data);
        data = NULL;
    }
    fclose(file);
    return data;
}

static int compare(const char *label, const char *name, const float *got,
                   const float *want, size_t count, double floor_cosine) {
    double dot = 0.0, got_norm = 0.0, want_norm = 0.0;
    double error = 0.0, scale = 0.0, worst = 0.0;
    size_t nonfinite = 0;
    for (size_t i = 0; i < count; i++) {
        double a = got[i], b = want[i];
        if (!isfinite(a)) { nonfinite++; continue; }
        dot += a * b;
        got_norm += a * a;
        want_norm += b * b;
        double difference = a - b;
        error += difference * difference;
        scale += b * b;
        if (fabs(difference) > worst) worst = fabs(difference);
    }
    double cosine = dot / (sqrt(got_norm) * sqrt(want_norm) + 1e-30);
    double relative = sqrt(error / (scale + 1e-30));
    int pass = cosine >= floor_cosine && nonfinite == 0;
    printf("%-4s %-9s cos=%.7f rel_l2=%.3e max_abs=%.3e nonfinite=%zu %s\n",
           name, label, cosine, relative, worst, nonfinite,
           pass ? "PASS" : "FAIL");
    return pass;
}

static int run(const char *label, const char *name, h3_ane_linear *linear,
               const float *x, uint32_t chunk_dim, uint32_t chunks,
               uint32_t rows, const float *reference, size_t outputs,
               double floor_cosine) {
    uint32_t plane_rows = h3_ane_linear_plane_rows(linear);
    size_t plane = (size_t)chunk_dim * rows;
    for (uint32_t c = 0; c < chunks; c++) {
        float *destination = h3_ane_linear_input(linear, c);
        for (uint32_t k = 0; k < chunk_dim; k++)
            memcpy(destination + (size_t)k * plane_rows,
                   x + (size_t)c * plane + (size_t)k * rows,
                   (size_t)rows * sizeof(float));
    }
    char error[256] = {0};
    if (!h3_ane_linear_eval(linear, error, sizeof(error))) {
        printf("%-4s %-9s eval failed: %s\n", name, label, error);
        return 0;
    }
    int pass = compare(label, name, h3_ane_linear_output(linear), reference,
                       outputs, floor_cosine);
    double best = 1e30;
    for (int i = 0; i < 5; i++) {
        double started = seconds();
        if (!h3_ane_linear_eval(linear, error, sizeof(error))) return 0;
        double elapsed = seconds() - started;
        if (elapsed < best) best = elapsed;
    }
    double work = 2.0 * (double)h3_ane_linear_output_dim(linear) *
                  chunk_dim * chunks * rows;
    printf("%-4s %-9s best_eval=%.3f ms  %.2f TFLOPS  compile=%.3f s  "
           "weights=%.1f MiB\n",
           name, label, best * 1e3, work / best / 1e12,
           h3_ane_linear_compile_seconds(linear),
           (double)h3_ane_linear_weight_bytes(linear) / (1024.0 * 1024.0));
    return pass;
}

int main(int argc, char **argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: %s NAME K N KC S WEIGHTS X YREF\n", argv[0]);
        return 2;
    }
    const char *name = argv[1];
    uint32_t input_dim = (uint32_t)atoi(argv[2]);
    uint32_t output_dim = (uint32_t)atoi(argv[3]);
    uint32_t chunk_dim = (uint32_t)atoi(argv[4]);
    uint32_t rows = (uint32_t)atoi(argv[5]);
    uint32_t chunks = (input_dim + chunk_dim - 1) / chunk_dim;
    size_t weight_bytes = (size_t)output_dim * chunk_dim * chunks * 2;
    size_t x_bytes = (size_t)chunk_dim * chunks * rows * sizeof(float);
    size_t y_bytes = (size_t)output_dim * rows * sizeof(float);

    uint16_t *weights = read_file(argv[6], weight_bytes);
    float *x = read_file(argv[7], x_bytes);
    float *reference = read_file(argv[8], y_bytes);
    if (!weights || !x || !reference) return 2;

    char error[256] = {0};
    int pass = 1;
    h3_ane_linear *linear = h3_ane_linear_create_chunked(
        name, weights, weight_bytes, input_dim, output_dim, rows, chunk_dim,
        error, sizeof(error));
    if (!linear) {
        printf("%-4s tiled     create failed: %s\n", name, error);
        return 1;
    }
    pass &= run("tiled", name, linear, x, chunk_dim, chunks, rows, reference,
                y_bytes / sizeof(float), 0.9999);
    h3_ane_linear_free(linear);

    /* Rebuild the row-major [N][K] weight the DiT holds, then let the module
     * tile it. This is the path the splice uses. */
    uint16_t *row_major = malloc((size_t)output_dim * input_dim * 2);
    if (!row_major) return 2;
    for (uint32_t n = 0; n < output_dim; n++)
        for (uint32_t c = 0; c < chunks; c++) {
            uint32_t base = c * chunk_dim;
            uint32_t span = input_dim > base ?
                (input_dim - base < chunk_dim ? input_dim - base : chunk_dim) : 0;
            memcpy(row_major + (size_t)n * input_dim + base,
                   weights + (size_t)c * output_dim * chunk_dim +
                       (size_t)n * chunk_dim,
                   (size_t)span * 2);
        }
    free(weights);
    linear = h3_ane_linear_create(name, row_major, H3_ANE_W_F16, input_dim,
                                  output_dim, rows, chunk_dim, error,
                                  sizeof(error));
    if (!linear) {
        printf("%-4s rowmajor  create failed: %s\n", name, error);
        return 1;
    }
    pass &= run("rowmajor", name, linear, x, chunk_dim, chunks, rows,
                reference, y_bytes / sizeof(float), 0.9999);
    h3_ane_linear_free(linear);
    free(row_major);
    free(x);
    free(reference);
    return pass ? 0 : 1;
}
