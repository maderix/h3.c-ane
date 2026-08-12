#ifndef H3_ANE_BRIDGE_H
#define H3_ANE_BRIDGE_H

#include <IOSurface/IOSurface.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The only interface to the private AppleNeuralEngine framework. Everything
 * above this file speaks MIL text, weight blobs, and IOSurfaces; everything
 * below it is objc_msgSend into _ANEInMemoryModel and friends.
 *
 * Binding rule: request inputs bind to MIL function parameters in
 * ALPHABETICAL name order, not declaration order. Callers must name their
 * parameters so the sort order matches the surface indices (i0_*, i1_*, or
 * x0..x7). Same-shape mismatches swap data silently. */

typedef struct h3_ane_model h3_ane_model;

int h3_ane_bridge_available(void);

/* 16KB-aligned, host-mapped, zero-fill on create is the caller's job. */
IOSurfaceRef h3_ane_bridge_surface(size_t bytes);

/* Compile (or restore from the compile cache) and load one MIL program.
 * Takes ownership of the malloc'd weights blob. The input surfaces bind to
 * indices 0..input_count-1 and `output` to output index 0. */
h3_ane_model *h3_ane_model_create(const char *name, const char *mil,
                                  void *weights, size_t weight_bytes,
                                  IOSurfaceRef *inputs, uint32_t input_count,
                                  IOSurfaceRef output,
                                  char *error, size_t error_size);
void h3_ane_model_free(h3_ane_model *model);

int h3_ane_model_eval(h3_ane_model *model, char *error, size_t error_size);

/* Residency rotation: unload releases the wired compiled net but keeps the
 * handle and request; reload re-wires it from the compile cache. */
int h3_ane_model_unload(h3_ane_model *model, char *error, size_t error_size);
int h3_ane_model_reload(h3_ane_model *model, char *error, size_t error_size);

/* Load cost of the create: a fresh compile or a cached load. */
double h3_ane_model_compile_seconds(const h3_ane_model *model);
bool h3_ane_model_cache_hit(const h3_ane_model *model);

/* Compiled models persist in $TMPDIR/h3-ane-cache/<content-hash> (the hash
 * covers the MIL text AND the weights, so same-shape different-weight models
 * never collide). Only the compiled artifacts are kept — the `data` file
 * embeds the constants. H3_ANE_CACHE=0 disables reuse, and freeing a model
 * while disabled also evicts its entry. */
bool h3_ane_cache_enabled(void);

#endif
