#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_weight_store h3_weight_store;

/* Open every safetensors header in a component directory without reading
 * tensor payloads. */
h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
void h3_weight_store_free(h3_weight_store *store);
size_t h3_weight_store_shards(const h3_weight_store *store);

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

/* Raw comfy-quants int8_tensorwise projection: caller frees quantized and
 * scales. convrot_group_size is 0 when the stored rows are unrotated. */
int h3_weight_load_int8_raw(const h3_weight_store *store, const char *name,
                            uint64_t rows, uint64_t columns,
                            int8_t **quantized, float **scales,
                            int *convrot_group_size,
                            char *error, size_t error_size);

/* Resolve an int8_tensorwise projection for SSD streaming: the payload
 * location plus the resident per-row scales (caller frees). The path aliases
 * the store's header and stays valid while the store is open. */
int h3_weight_int8_stream_source(const h3_weight_store *store,
                                 const char *name,
                                 uint64_t rows, uint64_t columns,
                                 const char **path, uint64_t *file_offset,
                                 float **scales, int *convrot_group_size,
                                 char *error, size_t error_size);

/* Stream an int8_tensorwise payload as dequantized+derotated BF16 row slabs.
 * Safe on an I/O thread; emit returns 0 to abort. */
int h3_weight_int8_stream_bf16(const char *path, uint64_t file_offset,
                               size_t rows, size_t columns,
                               const float *scales, int group_size,
                               int (*emit)(void *opaque, size_t row_begin,
                                           size_t row_count,
                                           const uint16_t *values),
                               void *opaque, char *error, size_t error_size);

#endif
