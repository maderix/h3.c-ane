#ifndef H3_CONVROT_H
#define H3_CONVROT_H

#include <stddef.h>
#include <stdint.h>

/* ConvRot: comfy-quants int8_tensorwise stores each weight row rotated per
 * group_size-wide K group by a normalized regular Hadamard matrix (the
 * Kronecker power of the no-all-ones H4, scaled 1/sqrt(size)). The matrix is
 * symmetric and orthogonal, so rotation and derotation are the same product. */

/* Cached row-major [size][size] table, or NULL if size is not a power of
 * four. */
const float *h3_convrot_hadamard(int size);

/* In-place derotation of a row-major [rows][columns] matrix: every group of
 * group_size values in a row is multiplied by H. Fails when columns is not a
 * multiple of group_size or the table is unavailable. */
int h3_convrot_derotate_f32(float *values, size_t rows, size_t columns,
                            int group_size);

#endif
