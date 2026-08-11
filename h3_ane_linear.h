#ifndef H3_ANE_LINEAR_H
#define H3_ANE_LINEAR_H

#include <stddef.h>
#include <stdint.h>

/* Neural Engine replacement for one DiT projection. The graph is a reduction
 * split: the K axis is cut into chunks of kc, every chunk is a 1x1 convolution
 * over a [1, kc, 1, rows] activation, and the partial products are summed.
 * Request input binding silently misbinds above eight inputs, so kc must keep
 * the chunk count at or below eight. */

typedef struct h3_ane_linear h3_ane_linear;

typedef enum {
    H3_ANE_W_F16 = 0,
    H3_ANE_W_BF16 = 1
} h3_ane_weight_dtype;

int h3_ane_linear_available(void);

/* Row-major [output_dim][input_dim] weights. */
h3_ane_linear *h3_ane_linear_create(const char *name, const void *weights,
                                    h3_ane_weight_dtype dtype,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);

/* Payload that is already chunks x [output_dim][kc] fp16 with the K padding
 * zeroed. Used by the fixture gate. */
h3_ane_linear *h3_ane_linear_create_chunked(const char *name,
                                    const void *chunks, size_t chunk_bytes,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);

void h3_ane_linear_free(h3_ane_linear *linear);

uint32_t h3_ane_linear_chunks(const h3_ane_linear *linear);
uint32_t h3_ane_linear_chunk_dim(const h3_ane_linear *linear);
uint32_t h3_ane_linear_rows(const h3_ane_linear *linear);
/* Row count the graph really runs: rows rounded up to a multiple of sixteen. */
uint32_t h3_ane_linear_plane_rows(const h3_ane_linear *linear);
uint32_t h3_ane_linear_output_dim(const h3_ane_linear *linear);

/* F32 [kc][rows] input plane for one chunk, and the F32 [output_dim][rows]
 * result. Both are IOSurface backed so a Metal kernel can write and read them
 * without a host copy. */
float *h3_ane_linear_input(h3_ane_linear *linear, uint32_t chunk);
float *h3_ane_linear_output(h3_ane_linear *linear);
size_t h3_ane_linear_input_bytes(const h3_ane_linear *linear);
size_t h3_ane_linear_output_bytes(const h3_ane_linear *linear);

int h3_ane_linear_eval(h3_ane_linear *linear, char *error, size_t error_size);

double h3_ane_linear_compile_seconds(const h3_ane_linear *linear);
uint64_t h3_ane_linear_weight_bytes(const h3_ane_linear *linear);

/* One spliced DiT projection: the ANE graph plus the Metal staging that moves
 * activations between h3.c's row-major BF16 rows and the channel-major F32
 * planes the Neural Engine reads. */

#include "h3_gpu.h"

typedef struct h3_ane_projection h3_ane_projection;

/* Largest reduction tile that keeps the graph at eight inputs or fewer. */
uint32_t h3_ane_linear_default_chunk(uint32_t input_dim);

h3_ane_projection *h3_ane_projection_create(h3_gpu *gpu, const char *name,
                                    const h3_gpu_tensor *weight,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size);
void h3_ane_projection_free(h3_ane_projection *projection);

/* Consumes the open command buffer, runs the Neural Engine, and leaves a new
 * command buffer open with the unpack encoded. */
int h3_ane_projection_apply(h3_ane_projection *projection, h3_gpu *gpu,
                            h3_gpu_tensor *output, const h3_gpu_tensor *input,
                            char *error, size_t error_size);

void h3_ane_projection_timings(const h3_ane_projection *projection,
                               double *pack_seconds, double *eval_seconds,
                               uint64_t *calls);
/* With H3_ANE_PROFILE_STAGES set, apply uses separate command buffers so sync,
 * pack and unpack can be measured without attributing prior Metal work to the
 * pack. This is a diagnostic mode and intentionally adds synchronization. */
void h3_ane_projection_stage_timings(const h3_ane_projection *projection,
                                     double *sync_seconds,
                                     double *pack_seconds,
                                     double *eval_seconds,
                                     double *unpack_seconds,
                                     uint64_t *calls);
uint64_t h3_ane_projection_weight_bytes(const h3_ane_projection *projection);
double h3_ane_projection_compile_seconds(const h3_ane_projection *projection);

#endif
