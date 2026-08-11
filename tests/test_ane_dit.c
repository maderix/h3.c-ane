/* Integration gate: one real h3_dit forward, pure Metal against the same
 * forward with H3_ANE_LINEARS=1.
 *
 * usage: h3_ane_dit_test [DIRECTORY] [ANE_BLOCKS]
 *
 * The released checkpoint does not fit on this machine, so the test mints a
 * synthetic DiT once and reuses it. Every block name points at the same tensor
 * payload, which keeps the file near 1.5 GiB and lets the loader share one page
 * cache across the 50 blocks. Everything after that is the real engine:
 * h3_dit_load_t2va, the schedule precompute, the token refiner, and
 * h3_dit_forward. */

#include "h3_dit.h"
#include "h3_host.h"
#include "h3_text_encoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

enum {
    HIDDEN = 5376,
    HEAD_DIM = 128,
    INNER = 7168,
    FFN = 14336,
    TEXT_DIM = 5120,
    TIME_DIM = 2688,
    TIME_INPUT = 256,
    BLOCK_OUTPUT = 3 * 6 * HIDDEN,
    FINAL_OUTPUT = 2 * HIDDEN,
    VIDEO_PATCH = 96,
    AUDIO_CHANNELS = 32,
    BLOCKS = 50,
    TEXT_TOKENS = 64
};

static double seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static float next_uniform(float scale) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return ((float)(rng_state >> 40) / 8388608.0f - 1.0f) * scale;
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* One unique payload. Many tensor names may reference the same entry. */
typedef struct {
    const char *dtype;
    uint64_t rows;
    uint64_t columns;   /* 0 for a rank-1 tensor */
    float scale;
    float centre;
    int rope;
    uint64_t offset;
    uint64_t bytes;
} payload;

enum {
    P_NORM_HIDDEN, P_QKV, P_HEAD_NORM, P_OUT, P_FC1, P_FC2, P_CONDITION,
    P_ADALN_W, P_ADALN_B, P_FINAL_ADALN_W, P_FINAL_ADALN_B,
    P_TIME_IN_W, P_TIME_IN_B, P_TIME_OUT_W, P_TIME_OUT_B,
    P_ROPE, P_VIDEO_PATCH_W, P_PATCH_B, P_AUDIO_PATCH_W,
    P_FINAL_VIDEO_W, P_FINAL_VIDEO_B, P_FINAL_AUDIO_W, P_FINAL_AUDIO_B,
    P_COUNT
};

static payload payloads[P_COUNT] = {
    [P_NORM_HIDDEN]    = {"BF16", HIDDEN, 0, 0.05f, 1.0f, 0, 0, 0},
    [P_QKV]            = {"BF16", 3 * INNER, HIDDEN, 0.02f, 0.0f, 0, 0, 0},
    [P_HEAD_NORM]      = {"BF16", HEAD_DIM, 0, 0.05f, 1.0f, 0, 0, 0},
    [P_OUT]            = {"BF16", HIDDEN, INNER, 0.02f, 0.0f, 0, 0, 0},
    [P_FC1]            = {"BF16", 2 * FFN, HIDDEN, 0.02f, 0.0f, 0, 0, 0},
    [P_FC2]            = {"BF16", HIDDEN, FFN, 0.02f, 0.0f, 0, 0, 0},
    [P_CONDITION]      = {"BF16", HIDDEN, TEXT_DIM, 0.02f, 0.0f, 0, 0, 0},
    [P_ADALN_W]        = {"BF16", BLOCK_OUTPUT, TIME_DIM, 0.005f, 0.0f, 0, 0, 0},
    [P_ADALN_B]        = {"BF16", BLOCK_OUTPUT, 0, 0.005f, 0.0f, 0, 0, 0},
    [P_FINAL_ADALN_W]  = {"BF16", FINAL_OUTPUT, TIME_DIM, 0.005f, 0.0f, 0, 0, 0},
    [P_FINAL_ADALN_B]  = {"BF16", FINAL_OUTPUT, 0, 0.005f, 0.0f, 0, 0, 0},
    [P_TIME_IN_W]      = {"F32", HIDDEN, TIME_INPUT, 0.02f, 0.0f, 0, 0, 0},
    [P_TIME_IN_B]      = {"F32", HIDDEN, 0, 0.01f, 0.0f, 0, 0, 0},
    [P_TIME_OUT_W]     = {"F32", TIME_DIM, HIDDEN, 0.02f, 0.0f, 0, 0, 0},
    [P_TIME_OUT_B]     = {"F32", TIME_DIM, 0, 0.01f, 0.0f, 0, 0, 0},
    [P_ROPE]           = {"F32", 16, 0, 0.0f, 0.0f, 1, 0, 0},
    [P_VIDEO_PATCH_W]  = {"F32", HIDDEN, VIDEO_PATCH, 0.05f, 0.0f, 0, 0, 0},
    [P_PATCH_B]        = {"F32", HIDDEN, 0, 0.01f, 0.0f, 0, 0, 0},
    [P_AUDIO_PATCH_W]  = {"F32", HIDDEN, AUDIO_CHANNELS, 0.05f, 0.0f, 0, 0, 0},
    [P_FINAL_VIDEO_W]  = {"F32", VIDEO_PATCH, HIDDEN, 0.02f, 0.0f, 0, 0, 0},
    [P_FINAL_VIDEO_B]  = {"F32", VIDEO_PATCH, 0, 0.01f, 0.0f, 0, 0, 0},
    [P_FINAL_AUDIO_W]  = {"F32", AUDIO_CHANNELS, HIDDEN, 0.02f, 0.0f, 0, 0, 0},
    [P_FINAL_AUDIO_B]  = {"F32", AUDIO_CHANNELS, 0, 0.01f, 0.0f, 0, 0, 0}
};

typedef struct {
    char name[96];
    int payload;
} entry;

static entry *entries = NULL;
static size_t entry_count = 0, entry_capacity = 0;

static void add(const char *format, int which, unsigned index) {
    if (entry_count == entry_capacity) {
        entry_capacity = entry_capacity ? entry_capacity * 2 : 128;
        entries = realloc(entries, entry_capacity * sizeof(*entries));
        if (!entries) abort();
    }
    snprintf(entries[entry_count].name, sizeof(entries[entry_count].name),
             format, index);
    entries[entry_count].payload = which;
    entry_count++;
}

static void add_block(const char *prefix) {
    char format[96];
    struct { const char *suffix; int which; } fields[8] = {
        {"norm1.weight", P_NORM_HIDDEN}, {"norm2.weight", P_NORM_HIDDEN},
        {"attn.qkv_proj.weight", P_QKV}, {"attn.q_norm.weight", P_HEAD_NORM},
        {"attn.k_norm.weight", P_HEAD_NORM}, {"attn.out_proj.weight", P_OUT},
        {"mlp.fc1.weight", P_FC1}, {"mlp.fc2.weight", P_FC2}
    };
    for (int i = 0; i < 8; i++) {
        snprintf(format, sizeof(format), "%s%s", prefix, fields[i].suffix);
        add(format, fields[i].which, 0);
    }
}

static void build_entries(void) {
    for (unsigned block = 0; block < BLOCKS; block++) {
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "blocks.%u.", block);
        add_block(prefix);
        char name[96];
        snprintf(name, sizeof(name), "blocks.%u.adaln_proj.linear.weight", block);
        add(name, P_ADALN_W, 0);
        snprintf(name, sizeof(name), "blocks.%u.adaln_proj.linear.bias", block);
        add(name, P_ADALN_B, 0);
    }
    add_block("token_refiner.blocks.0.");
    add_block("token_refiner.blocks.1.");
    add("token_refiner.final_norm.weight", P_NORM_HIDDEN, 0);
    add("condition_proj.weight", P_CONDITION, 0);
    add("condition_proj.bias", P_NORM_HIDDEN, 0);
    add("final_layer.adaln_proj.linear.weight", P_FINAL_ADALN_W, 0);
    add("final_layer.adaln_proj.linear.bias", P_FINAL_ADALN_B, 0);
    add("time_embedder.proj_in.weight", P_TIME_IN_W, 0);
    add("time_embedder.proj_in.bias", P_TIME_IN_B, 0);
    add("time_embedder.proj_out.weight", P_TIME_OUT_W, 0);
    add("time_embedder.proj_out.bias", P_TIME_OUT_B, 0);
    add("rope.inv_freq", P_ROPE, 0);
    add("video_patch_proj.weight", P_VIDEO_PATCH_W, 0);
    add("video_patch_proj.bias", P_PATCH_B, 0);
    add("audio_patch_proj.weight", P_AUDIO_PATCH_W, 0);
    add("audio_patch_proj.bias", P_PATCH_B, 0);
    add("final_layer.norm.weight", P_NORM_HIDDEN, 0);
    add("final_layer.video_out.weight", P_FINAL_VIDEO_W, 0);
    add("final_layer.video_out.bias", P_FINAL_VIDEO_B, 0);
    add("final_layer.audio_out.weight", P_FINAL_AUDIO_W, 0);
    add("final_layer.audio_out.bias", P_FINAL_AUDIO_B, 0);
}

static int write_payload(FILE *file, const payload *entry_payload) {
    uint64_t elements = entry_payload->rows *
        (entry_payload->columns ? entry_payload->columns : 1);
    int wide = !strcmp(entry_payload->dtype, "F32");
    size_t chunk = 1u << 16;
    void *buffer = malloc(chunk * (wide ? 4 : 2));
    if (!buffer) return 0;
    uint64_t written = 0;
    while (written < elements) {
        uint64_t count = elements - written < chunk ? elements - written : chunk;
        for (uint64_t i = 0; i < count; i++) {
            float value;
            if (entry_payload->rope)
                value = 1.0f / powf(10000.0f,
                                    (float)(written + i) * 2.0f / 32.0f);
            else
                value = entry_payload->centre + next_uniform(entry_payload->scale);
            if (wide) ((float *)buffer)[i] = value;
            else ((uint16_t *)buffer)[i] = to_bf16(value);
        }
        if (fwrite(buffer, wide ? 4 : 2, (size_t)count, file) != count) {
            free(buffer);
            return 0;
        }
        written += count;
    }
    free(buffer);
    return 1;
}

static int mint(const char *path) {
    FILE *file = fopen(path, "wb");
    if (!file) { fprintf(stderr, "cannot create %s\n", path); return 0; }
    uint64_t cursor = 0;
    for (int i = 0; i < P_COUNT; i++) {
        payload *p = &payloads[i];
        uint64_t elements = p->rows * (p->columns ? p->columns : 1);
        p->bytes = elements * (!strcmp(p->dtype, "F32") ? 4u : 2u);
        p->offset = cursor;
        cursor += p->bytes;
        cursor = (cursor + 63u) & ~UINT64_C(63);
    }
    size_t capacity = entry_count * 200 + 64;
    char *header = malloc(capacity);
    if (!header) { fclose(file); return 0; }
    size_t used = 0;
    used += (size_t)snprintf(header + used, capacity - used, "{");
    for (size_t i = 0; i < entry_count; i++) {
        const payload *p = &payloads[entries[i].payload];
        if (p->columns)
            used += (size_t)snprintf(header + used, capacity - used,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%llu,%llu],"
                "\"data_offsets\":[%llu,%llu]}", i ? "," : "", entries[i].name,
                p->dtype, (unsigned long long)p->rows,
                (unsigned long long)p->columns, (unsigned long long)p->offset,
                (unsigned long long)(p->offset + p->bytes));
        else
            used += (size_t)snprintf(header + used, capacity - used,
                "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[%llu],"
                "\"data_offsets\":[%llu,%llu]}", i ? "," : "", entries[i].name,
                p->dtype, (unsigned long long)p->rows,
                (unsigned long long)p->offset,
                (unsigned long long)(p->offset + p->bytes));
    }
    used += (size_t)snprintf(header + used, capacity - used, "}");
    size_t padded = (used + 63u) & ~(size_t)63u;
    uint64_t header_size = padded;
    if (fwrite(&header_size, 8, 1, file) != 1) { fclose(file); return 0; }
    if (fwrite(header, 1, used, file) != used) { fclose(file); return 0; }
    for (size_t i = used; i < padded; i++) fputc(' ', file);
    free(header);
    for (int i = 0; i < P_COUNT; i++) {
        long target = (long)(8 + padded + payloads[i].offset);
        if (fseek(file, target, SEEK_SET) != 0 ||
            !write_payload(file, &payloads[i])) {
            fprintf(stderr, "cannot write payload %d\n", i);
            fclose(file);
            return 0;
        }
    }
    fclose(file);
    return 1;
}

typedef struct {
    float *video;
    float *audio;
    size_t video_elements;
    size_t audio_elements;
    double load_seconds;
    double forward_seconds;
    uint32_t rows;
} run;

static unsigned active_blocks = BLOCKS;

static int forward_once(const char *directory, const h3_text_embedding *text,
                        const h3_layout *layout,
                        const h3_sigma_schedule *sigmas, run *out) {
    char error[512] = {0};
    double started = seconds();
    h3_dit *dit = h3_dit_load_t2va(directory, "h3_shaders.metal", text, layout,
                                   sigmas, active_blocks, 1, 0, 0, 1.0f,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   NULL, NULL, error, sizeof(error));
    if (!dit) {
        fprintf(stderr, "cannot load the DiT: %s\n", error);
        return 0;
    }
    out->load_seconds = seconds() - started;
    out->video_elements = h3_dit_video_elements(dit);
    out->audio_elements = h3_dit_audio_elements(dit);
    float *video = calloc(out->video_elements, sizeof(float));
    float *audio = calloc(out->audio_elements, sizeof(float));
    out->video = calloc(out->video_elements, sizeof(float));
    out->audio = calloc(out->audio_elements, sizeof(float));
    if (!video || !audio || !out->video || !out->audio) return 0;
    h3_rng rng;
    h3_rng_seed(&rng, 7);
    h3_rng_fill_normal(&rng, video, out->video_elements);
    h3_rng_fill_normal(&rng, audio, out->audio_elements);
    started = seconds();
    int ok = h3_dit_forward(dit, 0, video, audio, out->video, out->audio,
                            error, sizeof(error));
    out->forward_seconds = seconds() - started;
    if (!ok) fprintf(stderr, "forward failed: %s\n", error);
    free(video);
    free(audio);
    h3_dit_free(dit);
    return ok;
}

static double compare(const char *label, const float *got, const float *want,
                      size_t count) {
    double dot = 0.0, left = 0.0, right = 0.0, worst = 0.0;
    size_t nonfinite = 0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(got[i]) || !isfinite(want[i])) { nonfinite++; continue; }
        dot += (double)got[i] * want[i];
        left += (double)got[i] * got[i];
        right += (double)want[i] * want[i];
        double difference = fabs((double)got[i] - want[i]);
        if (difference > worst) worst = difference;
    }
    double cosine = dot / (sqrt(left) * sqrt(right) + 1e-30);
    printf("%-6s cos=%.6f max_abs=%.3e nonfinite=%zu elements=%zu\n",
           label, cosine, worst, nonfinite, count);
    return nonfinite ? -1.0 : cosine;
}

int main(int argc, char **argv) {
    const char *directory = argc > 1 ? argv[1] :
        "/Users/maderix/m4scratch/h3_synth_dit";
    const char *ane_blocks = argc > 2 ? argv[2] : "2";
    if (argc > 3) active_blocks = (unsigned)atoi(argv[3]);
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    build_entries();
    struct stat status;
    if (stat(path, &status) != 0) {
        mkdir(directory, 0755);
        printf("minting a synthetic DiT into %s\n", path);
        double started = seconds();
        if (!mint(path)) return 2;
        stat(path, &status);
        printf("minted %.2f GiB in %.1f s (%zu tensor names, %d payloads)\n",
               (double)status.st_size / (1024.0 * 1024.0 * 1024.0),
               seconds() - started, entry_count, P_COUNT);
    } else {
        printf("reusing %s (%.2f GiB)\n", path,
               (double)status.st_size / (1024.0 * 1024.0 * 1024.0));
    }

    h3_text_embedding text = {0};
    text.tokens = TEXT_TOKENS;
    text.width = TEXT_DIM;
    text.values = malloc((size_t)TEXT_TOKENS * TEXT_DIM * sizeof(uint16_t));
    if (!text.values) return 2;
    for (size_t i = 0; i < (size_t)TEXT_TOKENS * TEXT_DIM; i++)
        text.values[i] = to_bf16(next_uniform(1.0f));

    int latent_w = 0, latent_h = 0;
    h3_latent_canvas(512, 512, &latent_w, &latent_h);
    h3_temporal_shape temporal = h3_temporal(8);
    h3_layout_spec spec = {TEXT_TOKENS, temporal.video_t, latent_h, latent_w,
                           temporal.audio_t, temporal.frame_count,
                           NULL, 0, NULL, 0};
    h3_layout layout;
    char error[512] = {0};
    if (!h3_layout_build(&spec, &layout, error, sizeof(error))) {
        fprintf(stderr, "cannot build the layout: %s\n", error);
        return 2;
    }
    h3_sigma_schedule sigmas;
    if (!h3_serving_schedule_build(2, &sigmas)) return 2;
    printf("layout: text=%d latent=%dx%dx%d audio_t=%d sequence=%zu\n",
           TEXT_TOKENS, temporal.video_t, latent_h, latent_w, temporal.audio_t,
           layout.seq_len);

    run metal = {0}, ane = {0};
    unsetenv("H3_ANE_LINEARS");
    if (!forward_once(directory, &text, &layout, &sigmas, &metal)) return 1;
    setenv("H3_ANE_LINEARS", "1", 1);
    setenv("H3_ANE_LINEAR_BLOCKS", ane_blocks, 1);
    setenv("H3_PROFILE", "1", 1);
    if (!forward_once(directory, &text, &layout, &sigmas, &ane)) return 1;

    double video_cosine = compare("video", ane.video, metal.video,
                                  metal.video_elements);
    double audio_cosine = compare("audio", ane.audio, metal.audio,
                                  metal.audio_elements);
    printf("load  metal=%.1f s  ane=%.1f s\n", metal.load_seconds,
           ane.load_seconds);
    printf("forward metal=%.3f s  ane(%s of %u blocks)=%.3f s  "
           "delta per converted block=%.3f s\n",
           metal.forward_seconds, ane_blocks, active_blocks,
           ane.forward_seconds,
           (ane.forward_seconds - metal.forward_seconds) /
               (double)atoi(ane_blocks));
    int pass = video_cosine >= 0.999 && audio_cosine >= 0.999;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
