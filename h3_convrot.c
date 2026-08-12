#include "h3_convrot.h"

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define H3_CONVROT_MAX_TABLES 4
#define H3_CONVROT_SLAB_ROWS 8192

static pthread_mutex_t table_lock = PTHREAD_MUTEX_INITIALIZER;
static struct { int size; float *table; } tables[H3_CONVROT_MAX_TABLES];

static int power_of_four(int size) {
    if (size < 4 || (size & (size - 1))) return 0;
    while (size > 1) size >>= 2;
    return size == 1;
}

static float *build_table(int size) {
    static const int h4[4][4] = {
        {1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    float *current = malloc(sizeof(float) * size * size);
    float *next = malloc(sizeof(float) * size * size);
    if (!current || !next) { free(current); free(next); return NULL; }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) current[i * size + j] = (float)h4[i][j];
    int built = 4;
    while (built < size) {
        for (int bi = 0; bi < built; bi++)
            for (int bj = 0; bj < built; bj++)
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++)
                        next[(bi * 4 + i) * size + bj * 4 + j] =
                            current[bi * size + bj] * (float)h4[i][j];
        float *swap = current; current = next; next = swap;
        built *= 4;
    }
    free(next);
    float norm = 1.0f / sqrtf((float)size);
    for (int i = 0; i < size * size; i++) current[i] *= norm;
    return current;
}

const float *h3_convrot_hadamard(int size) {
    if (!power_of_four(size)) return NULL;
    pthread_mutex_lock(&table_lock);
    const float *result = NULL;
    for (int i = 0; i < H3_CONVROT_MAX_TABLES; i++) {
        if (tables[i].size == size) { result = tables[i].table; break; }
        if (!tables[i].size) {
            tables[i].table = build_table(size);
            if (tables[i].table) { tables[i].size = size; result = tables[i].table; }
            break;
        }
    }
    pthread_mutex_unlock(&table_lock);
    return result;
}

int h3_convrot_derotate_f32(float *values, size_t rows, size_t columns,
                            int group_size) {
    const float *table = h3_convrot_hadamard(group_size);
    if (!table || !values || !rows || !columns ||
        columns % (size_t)group_size) return 0;
    /* Row-major [rows][columns] with whole groups per row is one contiguous
     * [rows * groups][group_size] matrix; multiply slabs of it by H. */
    size_t group_rows = rows * (columns / (size_t)group_size);
    float *slab = malloc(sizeof(float) * H3_CONVROT_SLAB_ROWS * group_size);
    if (!slab) return 0;
    for (size_t begin = 0; begin < group_rows; begin += H3_CONVROT_SLAB_ROWS) {
        size_t count = group_rows - begin;
        if (count > H3_CONVROT_SLAB_ROWS) count = H3_CONVROT_SLAB_ROWS;
        float *source = values + begin * (size_t)group_size;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)count, group_size, group_size, 1.0f,
                    source, group_size, table, group_size, 0.0f,
                    slab, group_size);
        memcpy(source, slab, sizeof(float) * count * (size_t)group_size);
    }
    free(slab);
    return 1;
}
