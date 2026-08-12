#ifndef H3_ANE_BLOCK_H
#define H3_ANE_BLOCK_H

#include "h3_weights.h"

#include <stddef.h>
#include <stdint.h>

/* One full H3 DiT block as a single Neural Engine graph: adaln-modulated
 * RMSNorm, int8 ConvRot projections, per-head norm + 48-pair RoPE, full
 * softmax attention, gated residuals, and the SwiGLU MLP. Weights stay int8
 * on the engine; activations cross once per block.
 *
 * The packed sequence is described by up to H3_ANE_BLOCK_SEGMENTS contiguous
 * segments; each takes its own six modulation vectors from the mod input,
 * packed [segment][slot][hidden] with slots shift/scale/gate for attention
 * then shift/scale/gate for the MLP. */

#define H3_ANE_BLOCK_SEGMENTS 3
#define H3_ANE_BLOCK_SLOTS 6
/* mod input spatial width, padded to the ANE's 16-row binding rule */
#define H3_ANE_BLOCK_MOD_WIDTH 32

typedef struct h3_ane_block h3_ane_block;

/* Weights come straight from a comfy int8_tensorwise store (conventional
 * [q|k|v] rows are consumed natively). rope tables are [rows][96] with the
 * 48-pair halves duplicated. segment_ends holds the exclusive row bound of
 * every segment in sequence order. */
h3_ane_block *h3_ane_block_create(const h3_weight_store *store,
                                  const char *prefix, uint32_t rows,
                                  const uint32_t *segment_ends,
                                  const float *rope_cos, const float *rope_sin,
                                  char *error, size_t error_size);
void h3_ane_block_free(h3_ane_block *block);

/* F32 [5376][padded rows] channel-major activations. */
float *h3_ane_block_input(h3_ane_block *block);
/* F32 [5376][H3_ANE_BLOCK_MOD_WIDTH] mod plane; column seg*6+slot. */
float *h3_ane_block_mod(h3_ane_block *block);
float *h3_ane_block_output(h3_ane_block *block);
uint32_t h3_ane_block_padded_rows(const h3_ane_block *block);

int h3_ane_block_eval(h3_ane_block *block, char *error, size_t error_size);
double h3_ane_block_compile_seconds(const h3_ane_block *block);
uint64_t h3_ane_block_weight_bytes(const h3_ane_block *block);

#endif
