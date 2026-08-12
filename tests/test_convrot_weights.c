/* ConvRot int8 weight loading gates.
 *
 * The Hadamard table is pinned bit-exact to comfy-quants build_hadamard(256)
 * (Comfy-Org/comfy-quants src/comfy_quants/formats/convrot.py) via FNV-1a over
 * the fp32 little-endian bytes, so the table cannot drift from the format's
 * reference implementation. The loader gates run against a self-minted
 * safetensors fixture quantized with the reference formula in double
 * precision. */

#include "h3_convrot.h"
#include "h3_dit_schedule.h"
#include "h3_gpu.h"
#include "h3_text_encoder.h"
#include "h3_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GS 256
#define ROWS 8
#define COLS 512

static unsigned failures;

static void check(int ok, const char *label) {
    printf("%s %s\n", ok ? "ok" : "FAIL", label);
    if (!ok) failures++;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static double rng_uniform(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return ((double)((rng_state * 2685821657736338717ull) >> 11) /
            9007199254740992.0) - 0.5;
}

static uint16_t bf16_from_f32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static float f32_from_bf16(uint16_t half) {
    uint32_t bits = (uint32_t)half << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void hadamard_table_gates(void) {
    check(!h3_convrot_hadamard(128), "H(128) rejected (not a power of four)");
    check(!h3_convrot_hadamard(512), "H(512) rejected (not a power of four)");
    const float *table = h3_convrot_hadamard(GS);
    check(table != NULL, "H(256) built");
    if (!table) return;
    uint64_t hash = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < GS * GS; i++) {
        uint32_t bits;
        memcpy(&bits, &table[i], sizeof(bits));
        for (int b = 0; b < 4; b++) {
            hash ^= (bits >> (8 * b)) & 0xFF;
            hash *= 0x100000001b3ull;
        }
    }
    check(hash == 0x13cf444163fa2325ull,
          "H(256) is bit-exact to comfy-quants build_hadamard (FNV-1a)");
    double symmetry = 0, orthogonality = 0;
    for (int i = 0; i < GS; i++)
        for (int j = 0; j < GS; j++) {
            symmetry = fmax(symmetry,
                            fabs((double)table[i*GS+j] - table[j*GS+i]));
            double dot = 0;
            for (int k = 0; k < GS; k++)
                dot += (double)table[i*GS+k] * table[j*GS+k];
            orthogonality = fmax(orthogonality, fabs(dot - (i == j)));
        }
    check(symmetry == 0.0, "H(256) is symmetric");
    check(orthogonality < 1e-5, "H(256) is orthogonal");
}

static void derotate_gates(void) {
    const float *table = h3_convrot_hadamard(GS);
    enum { R = 13, C = 2 * GS };
    float *original = malloc(sizeof(float) * R * C);
    float *rotated = malloc(sizeof(float) * R * C);
    for (int i = 0; i < R * C; i++) original[i] = (float)rng_uniform();
    /* Offline rotation per the reference: W_rot = (W grouped) @ H.T. */
    for (int r = 0; r < R; r++)
        for (int g = 0; g < C / GS; g++)
            for (int i = 0; i < GS; i++) {
                double acc = 0;
                for (int j = 0; j < GS; j++)
                    acc += (double)original[r*C + g*GS + j] * table[i*GS + j];
                rotated[r*C + g*GS + i] = (float)acc;
            }
    check(h3_convrot_derotate_f32(rotated, R, C, GS),
          "derotate accepts [13][512]");
    double worst = 0;
    for (int i = 0; i < R * C; i++)
        worst = fmax(worst, fabs((double)rotated[i] - original[i]));
    check(worst < 1e-5, "derotation inverts the reference rotation");
    check(!h3_convrot_derotate_f32(rotated, R, C - 1, GS),
          "derotate rejects a non-multiple column count");
    free(original);
    free(rotated);
}

static size_t streamed_rows;
static int stream_collect(void *opaque, size_t row_begin, size_t row_count,
                          const uint16_t *values) {
    memcpy((uint16_t *)opaque + row_begin * COLS, values,
           row_count * COLS * sizeof(uint16_t));
    streamed_rows += row_count;
    return 1;
}

/* Minimal safetensors writer for the fixture. */
typedef struct {
    char json[4096];
    size_t json_length;
    uint8_t data[1 << 20];
    size_t data_length;
    int entries;
} fixture_t;

static void fixture_add(fixture_t *fixture, const char *name,
                        const char *dtype, const char *shape,
                        const void *payload, size_t bytes) {
    size_t begin = fixture->data_length;
    memcpy(fixture->data + begin, payload, bytes);
    fixture->data_length += bytes;
    fixture->json_length += (size_t)snprintf(
        fixture->json + fixture->json_length,
        sizeof(fixture->json) - fixture->json_length,
        "%s\"%s\":{\"dtype\":\"%s\",\"shape\":%s,\"data_offsets\":[%zu,%zu]}",
        fixture->entries++ ? "," : "{", name, dtype, shape, begin,
        fixture->data_length);
}

static int fixture_write(fixture_t *fixture, const char *path) {
    fixture->json_length += (size_t)snprintf(
        fixture->json + fixture->json_length,
        sizeof(fixture->json) - fixture->json_length, "}");
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    uint64_t header = fixture->json_length;
    int ok = fwrite(&header, 8, 1, file) == 1 &&
             fwrite(fixture->json, fixture->json_length, 1, file) == 1 &&
             fwrite(fixture->data, fixture->data_length, 1, file) == 1;
    fclose(file);
    return ok;
}

static void loader_gates(h3_gpu *gpu, const char *directory) {
    const float *table = h3_convrot_hadamard(GS);
    char error[512] = {0};

    /* Mint: W in double, rotate, per-row scale, round — the comfy formula. */
    static double original[ROWS][COLS];
    static double rotated[ROWS][COLS];
    static int8_t quantized[ROWS][COLS];
    static float scales[ROWS];
    for (int r = 0; r < ROWS; r++)
        for (int c = 0; c < COLS; c++) original[r][c] = rng_uniform();
    for (int r = 0; r < ROWS; r++)
        for (int g = 0; g < COLS / GS; g++)
            for (int i = 0; i < GS; i++) {
                double acc = 0;
                for (int j = 0; j < GS; j++)
                    acc += original[r][g*GS + j] * (double)table[i*GS + j];
                rotated[r][g*GS + i] = acc;
            }
    for (int r = 0; r < ROWS; r++) {
        double amax = 0;
        for (int c = 0; c < COLS; c++) amax = fmax(amax, fabs(rotated[r][c]));
        scales[r] = (float)(amax / 127.0);
        for (int c = 0; c < COLS; c++) {
            double q = round(rotated[r][c] / (double)scales[r]);
            quantized[r][c] = (int8_t)fmax(-128.0, fmin(127.0, q));
        }
    }
    static const char marker[] =
        "{\"format\": \"int8_tensorwise\", \"convrot\": true, "
        "\"convrot_groupsize\": 256}";
    static const char bad_marker[] =
        "{\"format\": \"fp8_rowwise\", \"convrot\": true, "
        "\"convrot_groupsize\": 256}";
    static uint16_t plain[4][8];
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++)
            plain[r][c] = bf16_from_f32((float)(r * 8 + c) * 0.25f - 3.0f);
    static _Float16 half[16];
    for (int i = 0; i < 16; i++) half[i] = (_Float16)(0.125 * i - 1.0);

    char marker_shape[16], bad_marker_shape[16];
    snprintf(marker_shape, sizeof(marker_shape), "[%zu]", sizeof(marker) - 1);
    snprintf(bad_marker_shape, sizeof(bad_marker_shape), "[%zu]",
             sizeof(bad_marker) - 1);
    fixture_t fixture = {0};
    fixture_add(&fixture, "proj.weight", "I8", "[8,512]",
                quantized, sizeof(quantized));
    fixture_add(&fixture, "proj.weight_scale", "F32", "[8,1]",
                scales, sizeof(scales));
    fixture_add(&fixture, "proj.comfy_quant", "U8", marker_shape,
                marker, sizeof(marker) - 1);
    fixture_add(&fixture, "bad.weight", "I8", "[8,512]",
                quantized, sizeof(quantized));
    fixture_add(&fixture, "bad.weight_scale", "F32", "[8,1]",
                scales, sizeof(scales));
    fixture_add(&fixture, "bad.comfy_quant", "U8", bad_marker_shape,
                bad_marker, sizeof(bad_marker) - 1);
    fixture_add(&fixture, "plain.weight", "BF16", "[4,8]",
                plain, sizeof(plain));
    fixture_add(&fixture, "half.weight", "F16", "[16]",
                half, sizeof(half));
    char path[512];
    snprintf(path, sizeof(path), "%s/convrot_fixture.safetensors", directory);
    check(fixture_write(&fixture, path), "fixture written");

    h3_weight_store *store = h3_weight_store_open(directory, error,
                                                  sizeof(error));
    check(store != NULL, "weight store opens the fixture");
    if (!store) { printf("  %s\n", error); return; }

    /* int8 -> BF16 through the real loader. */
    uint64_t shape2[] = {ROWS, COLS};
    h3_gpu_tensor *tensor = h3_weight_load_bf16(store, gpu, "proj.weight", 2,
                                                shape2, error, sizeof(error));
    check(tensor != NULL, "int8 projection loads as BF16");
    if (tensor) {
        static uint16_t loaded[ROWS * COLS];
        check(h3_gpu_tensor_read_bf16(tensor, loaded, ROWS * COLS),
              "loaded projection reads back");
        double worst = 0, reference_peak = 0;
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++) {
                /* Independent double reference: dequantize, derotate. */
                double expected = 0;
                int g = c / GS, j = c % GS;
                for (int i = 0; i < GS; i++)
                    expected += (double)quantized[r][g*GS + i] *
                                (double)scales[r] * (double)table[i*GS + j];
                double got = f32_from_bf16(loaded[r*COLS + c]);
                worst = fmax(worst, fabs(got - expected));
                reference_peak = fmax(reference_peak, fabs(expected));
            }
        /* BF16 storage grants ~2^-8 relative; orientation errors are O(1). */
        check(worst <= reference_peak * 0.02,
              "dequantized+derotated values match the double reference");
        double drift = 0;
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++)
                drift = fmax(drift, fabs(f32_from_bf16(loaded[r*COLS + c]) -
                                         original[r][c]));
        check(drift < 0.02, "loaded values approximate the pre-quant weights");
        h3_gpu_tensor_free(tensor);
    } else printf("  %s\n", error);

    /* Raw int8 access for the ANE path. */
    int8_t *raw = NULL;
    float *raw_scales = NULL;
    int group_size = -1;
    check(h3_weight_load_int8_raw(store, "proj.weight", ROWS, COLS, &raw,
                                  &raw_scales, &group_size, error,
                                  sizeof(error)),
          "raw int8 load succeeds");
    if (raw) {
        check(group_size == GS, "raw load reports the convrot group size");
        check(!memcmp(raw, quantized, sizeof(quantized)),
              "raw int8 bytes are unmodified");
        check(!memcmp(raw_scales, scales, sizeof(scales)),
              "raw scales are unmodified");
    }
    free(raw);
    free(raw_scales);

    /* Contract violations fail loudly. */
    error[0] = '\0';
    tensor = h3_weight_load_bf16(store, gpu, "bad.weight", 2, shape2, error,
                                 sizeof(error));
    check(!tensor && strstr(error, "unsupported quant format"),
          "unknown quant format is rejected, not decoded");
    h3_gpu_tensor_free(tensor);
    error[0] = '\0';
    uint64_t wrong[] = {ROWS, COLS + 1};
    tensor = h3_weight_load_bf16(store, gpu, "proj.weight", 2, wrong, error,
                                 sizeof(error));
    check(!tensor, "shape mismatch is rejected");
    h3_gpu_tensor_free(tensor);

    /* SSD-streaming form of the same projection: source resolution plus
     * slabbed dequant+derotate must agree with the double reference. */
    {
        const char *stream_path = NULL;
        uint64_t stream_offset = 0;
        float *stream_scales = NULL;
        int stream_gs = 0;
        check(h3_weight_int8_stream_source(store, "proj.weight", ROWS, COLS,
                                           &stream_path, &stream_offset,
                                           &stream_scales, &stream_gs, error,
                                           sizeof(error)),
              "int8 stream source resolves");
        static uint16_t streamed[ROWS * COLS];
        streamed_rows = 0;
        check(stream_path &&
              h3_weight_int8_stream_bf16(stream_path, stream_offset, ROWS,
                                         COLS, stream_scales, stream_gs,
                                         stream_collect, streamed, error,
                                         sizeof(error)),
              "int8 stream reads all slabs");
        check(streamed_rows == ROWS, "int8 stream visits every row");
        double worst = 0, peak = 0;
        for (int r = 0; r < ROWS; r++)
            for (int c = 0; c < COLS; c++) {
                int g = c / GS, j = c % GS;
                double expected = 0;
                for (int i = 0; i < GS; i++)
                    expected += (double)quantized[r][g*GS + i] *
                                (double)scales[r] * (double)table[i*GS + j];
                worst = fmax(worst, fabs(f32_from_bf16(streamed[r*COLS + c]) -
                                         expected));
                peak = fmax(peak, fabs(expected));
            }
        check(worst <= peak * 0.02,
              "streamed rows match the double reference");
        free(stream_scales);
    }

    /* BF16 fast path is unchanged; F16 converts. */
    uint64_t plain_shape[] = {4, 8};
    tensor = h3_weight_load_bf16(store, gpu, "plain.weight", 2, plain_shape,
                                 error, sizeof(error));
    check(tensor != NULL, "BF16 weight loads");
    if (tensor) {
        uint16_t loaded[32];
        h3_gpu_tensor_read_bf16(tensor, loaded, 32);
        check(!memcmp(loaded, plain, sizeof(plain)), "BF16 payload is exact");
        h3_gpu_tensor_free(tensor);
    }
    uint64_t half_shape[] = {16};
    tensor = h3_weight_load_bf16(store, gpu, "half.weight", 1, half_shape,
                                 error, sizeof(error));
    check(tensor != NULL, "F16 weight loads as BF16");
    if (tensor) {
        uint16_t loaded[16];
        h3_gpu_tensor_read_bf16(tensor, loaded, 16);
        int exact = 1;
        for (int i = 0; i < 16; i++)
            if (loaded[i] != bf16_from_f32((float)half[i])) exact = 0;
        check(exact, "F16 conversion rounds to nearest-even BF16");
        h3_gpu_tensor_free(tensor);
    }
    tensor = h3_weight_load_f32(store, gpu, "half.weight", 1, half_shape,
                                error, sizeof(error));
    check(tensor != NULL, "F16 weight loads as F32");
    if (tensor) {
        float loaded[16];
        h3_gpu_tensor_read_f32(tensor, loaded, 16);
        int exact = 1;
        for (int i = 0; i < 16; i++)
            if (loaded[i] != (float)half[i]) exact = 0;
        check(exact, "F16 to F32 widening is exact");
        h3_gpu_tensor_free(tensor);
    }

    h3_weight_store_free(store);
    unlink(path);
}

static void conditioning_gates(const char *directory) {
    char error[512] = {0};
    enum { TOKENS = 3 };
    static uint16_t values[TOKENS * H3_TEXT_HIDDEN_SIZE];
    static uint8_t tags[TOKENS] = {1, 0, 1};
    for (size_t i = 0; i < TOKENS * H3_TEXT_HIDDEN_SIZE; i++)
        values[i] = (uint16_t)(0x3f80 + (i % 251));
    h3_text_embedding source = {TOKENS, H3_TEXT_HIDDEN_SIZE, values, {0}, tags};
    char path[512];
    snprintf(path, sizeof(path), "%s/conditioning.h3cd", directory);
    check(h3_conditioning_file_write(path, &source, error, sizeof(error)),
          "conditioning file writes");
    h3_text_embedding loaded = {0};
    check(h3_conditioning_file_read(path, &loaded, error, sizeof(error)),
          "conditioning file reads back");
    check(loaded.tokens == TOKENS && loaded.width == H3_TEXT_HIDDEN_SIZE &&
          loaded.values && loaded.tags &&
          !memcmp(loaded.values, values, sizeof(values)) &&
          !memcmp(loaded.tags, tags, sizeof(tags)),
          "conditioning round-trip is exact");
    h3_text_embedding_free(&loaded);
    /* A corrupt header must be rejected, not decoded. */
    FILE *file = fopen(path, "r+b");
    if (file) {
        uint32_t bad = 0;
        fwrite(&bad, sizeof(bad), 1, file);
        fclose(file);
    }
    error[0] = '\0';
    check(!h3_conditioning_file_read(path, &loaded, error, sizeof(error)) &&
          strstr(error, "H3CD"), "corrupt conditioning file is rejected");
    unlink(path);
}

/* With a real int8 checkpoint directory as argv[1]: load one full projection
 * through the production path and cross-check a sampled row against an
 * independent double dequant+derotate of the raw payload. */
static void real_checkpoint_gate(h3_gpu *gpu, const char *directory) {
    char error[512] = {0};
    h3_weight_store *store = h3_weight_store_open(directory, error,
                                                  sizeof(error));
    check(store != NULL, "real checkpoint store opens");
    if (!store) { printf("  %s\n", error); return; }
    const char *name = "blocks.0.attn.qkv_proj.weight";
    enum { RQ = 21504, CQ = 5376 };
    int8_t *raw = NULL;
    float *raw_scales = NULL;
    int group_size = 0;
    check(h3_weight_load_int8_raw(store, name, RQ, CQ, &raw, &raw_scales,
                                  &group_size, error, sizeof(error)),
          "real raw int8 load");
    uint64_t shape[] = {RQ, CQ};
    h3_gpu_tensor *tensor = h3_weight_load_bf16(store, gpu, name, 2, shape,
                                                error, sizeof(error));
    check(tensor != NULL, "real projection loads as BF16");
    if (tensor && raw && group_size == GS) {
        const float *table = h3_convrot_hadamard(GS);
        static uint16_t row[CQ];
        check(h3_gpu_tensor_read_bf16(tensor, row, CQ),
              "real projection row reads back");
        double worst = 0, peak = 0;
        for (int c = 0; c < CQ; c++) {
            int g = c / GS, j = c % GS;
            double expected = 0;
            for (int i = 0; i < GS; i++)
                expected += (double)raw[g*GS + i] * (double)raw_scales[0] *
                            (double)table[i*GS + j];
            double got = f32_from_bf16(row[c]);
            worst = fmax(worst, fabs(got - expected));
            peak = fmax(peak, fabs(expected));
        }
        check(worst <= peak * 0.02,
              "real projection row 0 matches the double reference");
        printf("info real qkv worst=%.3e peak=%.3e\n", worst, peak);
    } else if (!tensor) printf("  %s\n", error);
    h3_gpu_tensor_free(tensor);
    free(raw);
    free(raw_scales);
    h3_weight_store_free(store);
}

/* Adaln-curve lerp gate: a tiny table fixture, times spanning the clamp and
 * interior cases, checked against the ComfyUI reference formula
 * lerp(table[floor(t*(grid-1))], table[floor+1], frac) with edge clamping. */
static void curve_gates(h3_gpu *gpu, const char *directory) {
    char error[512] = {0};
    enum { GRID = 5, WIDTH = 4 };
    static float table[GRID][WIDTH];
    for (int g = 0; g < GRID; g++)
        for (int w = 0; w < WIDTH; w++)
            table[g][w] = (float)(g * g) * 0.25f + (float)w * 0.125f;
    fixture_t fixture = {0};
    fixture_add(&fixture, "adaln_t_table", "F32", "[5,4]",
                table, sizeof(table));
    char path[512];
    snprintf(path, sizeof(path), "%s/curve_fixture.safetensors", directory);
    check(fixture_write(&fixture, path), "curve fixture written");
    h3_weight_store *store = h3_weight_store_open(directory, error,
                                                  sizeof(error));
    check(store != NULL, "curve store opens");
    if (!store) return;
    float times[5] = {-0.5f, 0.0f, 0.30f, 0.99f, 1.5f};
    uint32_t width = 0;
    h3_gpu_tensor *rows = h3_dit_time_curve_embeddings(
        store, gpu, 5, times, &width, error, sizeof(error));
    check(rows != NULL && width == WIDTH, "curve rows build at table width");
    if (rows) {
        uint16_t loaded[5 * WIDTH];
        h3_gpu_tensor_read_bf16(rows, loaded, 5 * WIDTH);
        double worst = 0;
        for (int row = 0; row < 5; row++) {
            float t = times[row] < 0 ? 0 : (times[row] > 1 ? 1.0f : times[row]);
            float position = t * (GRID - 1);
            int low = (int)position;
            if (low > GRID - 2) low = GRID - 2;
            float blend = position - (float)low;
            for (int w = 0; w < WIDTH; w++) {
                float want = table[low][w] +
                             (table[low + 1][w] - table[low][w]) * blend;
                float got = f32_from_bf16(loaded[row * WIDTH + w]);
                worst = fmax(worst, fabs((double)got - want));
            }
        }
        check(worst <= 0.02, "curve lerp and clamping match the reference");
        h3_gpu_tensor_free(rows);
    } else printf("  %s\n", error);
    h3_weight_store_free(store);
    unlink(path);
}

int main(int argc, char **argv) {
    hadamard_table_gates();
    derotate_gates();
    char error[512] = {0};
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) {
        printf("FAIL gpu: %s\n", error);
        return 1;
    }
    char directory[] = "/tmp/h3_convrot_test.XXXXXX";
    if (!mkdtemp(directory)) {
        printf("FAIL mkdtemp\n");
        return 1;
    }
    loader_gates(gpu, directory);
    conditioning_gates(directory);
    curve_gates(gpu, directory);
    rmdir(directory);
    if (argc > 1) real_checkpoint_gate(gpu, argv[1]);
    h3_gpu_free(gpu);
    printf("%u failure(s)\n", failures);
    return failures ? 1 : 0;
}
