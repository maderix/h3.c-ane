#include "h3_weights.h"

#include "h3_convrot.h"

#include <dirent.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_weight_store {
    h3_st_header *headers;
    size_t count;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int safetensors_name(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

static void free_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++) free(paths[index]);
    free(paths);
}

h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size) {
    if (!directory || !*directory) {
        fail(error, error_size, "weight directory is required");
        return NULL;
    }
    DIR *stream = opendir(directory);
    if (!stream) {
        fail(error, error_size, "cannot open weight directory: %s", directory);
        return NULL;
    }
    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (!safetensors_name(entry->d_name)) continue;
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            char **grown = realloc(paths, next * sizeof(*grown));
            if (!grown) {
                fail(error, error_size, "out of memory listing weight shards");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            paths = grown;
            capacity = next;
        }
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        paths[count] = malloc(length);
        if (!paths[count]) {
            fail(error, error_size, "out of memory resolving a weight shard");
            closedir(stream);
            free_paths(paths, count);
            return NULL;
        }
        snprintf(paths[count], length, "%s/%s", directory, entry->d_name);
        count++;
    }
    closedir(stream);
    if (!count) {
        fail(error, error_size, "no safetensors shards in %s", directory);
        free(paths);
        return NULL;
    }
    qsort(paths, count, sizeof(*paths), compare_paths);
    h3_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory creating weight store");
        free_paths(paths, count);
        return NULL;
    }
    store->headers = calloc(count, sizeof(*store->headers));
    if (!store->headers) {
        fail(error, error_size, "out of memory allocating weight headers");
        free(store);
        free_paths(paths, count);
        return NULL;
    }
    store->count = count;
    for (size_t index = 0; index < count; index++) {
        char detail[384];
        if (!h3_st_read_header(paths[index], &store->headers[index], detail,
                               sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            free_paths(paths, count);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    free_paths(paths, count);
    return store;
}

void h3_weight_store_free(h3_weight_store *store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++) {
        h3_st_free_header(&store->headers[index]);
    }
    free(store->headers);
    free(store);
}

size_t h3_weight_store_shards(const h3_weight_store *store) {
    return store ? store->count : 0;
}

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
    for (size_t index = 0; index < store->count; index++) {
        const h3_st_tensor *tensor = h3_st_find(&store->headers[index], name);
        if (tensor) {
            if (header) *header = &store->headers[index];
            return tensor;
        }
    }
    return NULL;
}

static uint16_t bf16_from_f32(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* Sidecar name: "<base>.weight" -> "<base>.<suffix>". */
static int sidecar_name(const char *name, const char *suffix,
                        char *out, size_t out_size) {
    static const char weight[] = ".weight";
    size_t length = strlen(name);
    if (length <= sizeof(weight) - 1 ||
        strcmp(name + length - (sizeof(weight) - 1), weight) != 0) return 0;
    size_t base = length - (sizeof(weight) - 1);
    if (base + 1 + strlen(suffix) + 1 > out_size) return 0;
    memcpy(out, name, base);
    out[base] = '.';
    strcpy(out + base + 1, suffix);
    return 1;
}

/* comfy_quant sidecar: require int8_tensorwise; group size when rotated,
 * 0 when the marker says unrotated, -1 on any contract violation. */
static int comfy_quant_group_size(const h3_weight_store *store,
                                  const char *weight_name,
                                  char *error, size_t error_size) {
    char name[192];
    if (!sidecar_name(weight_name, "comfy_quant", name, sizeof(name))) {
        fail(error, error_size, "cannot derive comfy_quant name for %s",
             weight_name);
        return -1;
    }
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor || tensor->dtype != H3_DTYPE_U8) {
        fail(error, error_size, "int8 weight %s has no comfy_quant marker",
             weight_name);
        return -1;
    }
    uint64_t bytes = h3_st_tensor_elements(tensor);
    if (!bytes || bytes > 512) {
        fail(error, error_size, "comfy_quant marker for %s has %llu bytes",
             weight_name, (unsigned long long)bytes);
        return -1;
    }
    char marker[513];
    char detail[384];
    if (!h3_st_read_data(header, tensor, marker, (size_t)bytes, detail,
                         sizeof(detail))) {
        fail(error, error_size, "%s", detail);
        return -1;
    }
    marker[bytes] = '\0';
    if (!strstr(marker, "\"format\": \"int8_tensorwise\"")) {
        fail(error, error_size, "weight %s uses an unsupported quant format: %s",
             weight_name, marker);
        return -1;
    }
    if (!strstr(marker, "\"convrot\": true")) return 0;
    const char *field = strstr(marker, "\"convrot_groupsize\":");
    int group_size = field ? atoi(field + strlen("\"convrot_groupsize\":")) : 0;
    if (group_size <= 0 || !h3_convrot_hadamard(group_size)) {
        fail(error, error_size, "weight %s has an unusable convrot group size",
             weight_name);
        return -1;
    }
    return group_size;
}

static float *read_int8_scales(const h3_weight_store *store,
                               const char *weight_name, uint64_t rows,
                               char *error, size_t error_size) {
    char name[192];
    if (!sidecar_name(weight_name, "weight_scale", name, sizeof(name))) {
        fail(error, error_size, "cannot derive weight_scale name for %s",
             weight_name);
        return NULL;
    }
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor || tensor->dtype != H3_DTYPE_F32 ||
        h3_st_tensor_elements(tensor) != rows) {
        fail(error, error_size, "int8 weight %s needs F32 weight_scale[%llu]",
             weight_name, (unsigned long long)rows);
        return NULL;
    }
    float *scales = malloc(sizeof(float) * rows);
    if (!scales) {
        fail(error, error_size, "out of memory reading scales for %s",
             weight_name);
        return NULL;
    }
    char detail[384];
    if (!h3_st_read_data(header, tensor, scales, sizeof(float) * rows,
                         detail, sizeof(detail))) {
        fail(error, error_size, "%s", detail);
        free(scales);
        return NULL;
    }
    return scales;
}

int h3_weight_load_int8_raw(const h3_weight_store *store, const char *name,
                            uint64_t rows, uint64_t columns,
                            int8_t **quantized, float **scales,
                            int *convrot_group_size,
                            char *error, size_t error_size) {
    *quantized = NULL;
    *scales = NULL;
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor || tensor->dtype != H3_DTYPE_I8 || tensor->ndim != 2 ||
        tensor->shape[0] != rows || tensor->shape[1] != columns) {
        fail(error, error_size, "int8 weight %s is not I8 [%llu][%llu]", name,
             (unsigned long long)rows, (unsigned long long)columns);
        return 0;
    }
    int group_size = comfy_quant_group_size(store, name, error, error_size);
    if (group_size < 0) return 0;
    if (group_size && columns % (uint64_t)group_size) {
        fail(error, error_size, "weight %s columns are not a convrot multiple",
             name);
        return 0;
    }
    float *row_scales = read_int8_scales(store, name, rows, error, error_size);
    if (!row_scales) return 0;
    size_t bytes = (size_t)rows * columns;
    int8_t *data = malloc(bytes);
    char detail[384];
    if (!data || !h3_st_read_data(header, tensor, data, bytes, detail,
                                  sizeof(detail))) {
        fail(error, error_size, "%s",
             data ? detail : "out of memory reading an int8 weight");
        free(data);
        free(row_scales);
        return 0;
    }
    *quantized = data;
    *scales = row_scales;
    *convrot_group_size = group_size;
    return 1;
}

int h3_weight_int8_stream_source(const h3_weight_store *store,
                                 const char *name,
                                 uint64_t rows, uint64_t columns,
                                 const char **path, uint64_t *file_offset,
                                 float **scales, int *convrot_group_size,
                                 char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor || tensor->dtype != H3_DTYPE_I8 || tensor->ndim != 2 ||
        tensor->shape[0] != rows || tensor->shape[1] != columns) {
        fail(error, error_size, "int8 stream weight %s is not I8 [%llu][%llu]",
             name, (unsigned long long)rows, (unsigned long long)columns);
        return 0;
    }
    int group_size = comfy_quant_group_size(store, name, error, error_size);
    if (group_size < 0) return 0;
    if (group_size && columns % (uint64_t)group_size) {
        fail(error, error_size, "weight %s columns are not a convrot multiple",
             name);
        return 0;
    }
    float *row_scales = read_int8_scales(store, name, rows, error, error_size);
    if (!row_scales) return 0;
    *path = header->path;
    *file_offset = tensor->file_offset;
    *scales = row_scales;
    *convrot_group_size = group_size;
    return 1;
}

#define H3_INT8_STREAM_ROWS 2048

int h3_weight_int8_stream_bf16(const char *path, uint64_t file_offset,
                               size_t rows, size_t columns,
                               const float *scales, int group_size,
                               int (*emit)(void *opaque, size_t row_begin,
                                           size_t row_count,
                                           const uint16_t *values),
                               void *opaque, char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fail(error, error_size, "cannot open %s for int8 streaming", path);
        return 0;
    }
    size_t slab_rows = rows < H3_INT8_STREAM_ROWS ? rows : H3_INT8_STREAM_ROWS;
    int8_t *quantized = malloc(slab_rows * columns);
    float *wide = malloc(sizeof(float) * slab_rows * columns);
    uint16_t *narrow = malloc(sizeof(uint16_t) * slab_rows * columns);
    int ok = quantized && wide && narrow;
    if (!ok) fail(error, error_size, "out of memory streaming an int8 weight");
    for (size_t begin = 0; ok && begin < rows; begin += slab_rows) {
        size_t count = rows - begin < slab_rows ? rows - begin : slab_rows;
        if (fseeko(file, (off_t)(file_offset + begin * columns), SEEK_SET) ||
            fread(quantized, 1, count * columns, file) != count * columns) {
            fail(error, error_size, "cannot read int8 rows from %s", path);
            ok = 0;
            break;
        }
        for (size_t row = 0; row < count; row++) {
            float scale = scales[begin + row];
            const int8_t *source = quantized + row * columns;
            float *destination = wide + row * columns;
            for (size_t column = 0; column < columns; column++)
                destination[column] = (float)source[column] * scale;
        }
        if (group_size &&
            !h3_convrot_derotate_f32(wide, count, columns, group_size)) {
            fail(error, error_size, "cannot derotate streamed int8 rows");
            ok = 0;
            break;
        }
        for (size_t index = 0; index < count * columns; index++)
            narrow[index] = bf16_from_f32(wide[index]);
        if (!emit(opaque, begin, count, narrow)) {
            fail(error, error_size, "int8 stream destination rejected rows");
            ok = 0;
        }
    }
    free(quantized);
    free(wide);
    free(narrow);
    fclose(file);
    return ok;
}

/* Dequantize (+ derotate) an int8_tensorwise projection to BF16. */
static h3_gpu_tensor *load_int8_as_bf16(const h3_weight_store *store,
                                        h3_gpu *gpu, const char *name,
                                        uint64_t rows, uint64_t columns,
                                        char *error, size_t error_size) {
    int8_t *quantized = NULL;
    float *scales = NULL;
    int group_size = 0;
    if (!h3_weight_load_int8_raw(store, name, rows, columns, &quantized,
                                 &scales, &group_size, error, error_size))
        return NULL;
    size_t elements = (size_t)rows * columns;
    float *wide = malloc(sizeof(float) * elements);
    uint16_t *narrow = malloc(sizeof(uint16_t) * elements);
    h3_gpu_tensor *result = NULL;
    if (wide && narrow) {
        for (uint64_t row = 0; row < rows; row++) {
            float scale = scales[row];
            const int8_t *source = quantized + row * columns;
            float *destination = wide + row * columns;
            for (uint64_t column = 0; column < columns; column++)
                destination[column] = (float)source[column] * scale;
        }
        if (!group_size ||
            h3_convrot_derotate_f32(wide, rows, columns, group_size)) {
            for (size_t index = 0; index < elements; index++)
                narrow[index] = bf16_from_f32(wide[index]);
            result = h3_gpu_tensor_from_bf16(gpu, narrow, elements);
            if (!result)
                fail(error, error_size, "cannot upload %s: %s", name,
                     h3_gpu_error(gpu));
        } else {
            fail(error, error_size, "cannot derotate int8 weight %s", name);
        }
    } else {
        fail(error, error_size, "out of memory dequantizing %s", name);
    }
    free(wide);
    free(narrow);
    free(quantized);
    free(scales);
    return result;
}

static h3_gpu_tensor *load_f16_as_bf16(const h3_st_header *header,
                                       const h3_st_tensor *tensor,
                                       h3_gpu *gpu, const char *name,
                                       size_t elements,
                                       char *error, size_t error_size) {
    _Float16 *raw = malloc(sizeof(_Float16) * elements);
    uint16_t *narrow = malloc(sizeof(uint16_t) * elements);
    h3_gpu_tensor *result = NULL;
    char detail[384];
    if (raw && narrow &&
        h3_st_read_data(header, tensor, raw, sizeof(_Float16) * elements,
                        detail, sizeof(detail))) {
        for (size_t index = 0; index < elements; index++)
            narrow[index] = bf16_from_f32((float)raw[index]);
        result = h3_gpu_tensor_from_bf16(gpu, narrow, elements);
        if (!result)
            fail(error, error_size, "cannot upload %s: %s", name,
                 h3_gpu_error(gpu));
    } else {
        fail(error, error_size, "%s",
             raw && narrow ? detail : "out of memory converting an F16 weight");
    }
    free(raw);
    free(narrow);
    return result;
}

static h3_gpu_tensor *load_f16_as_f32(const h3_st_header *header,
                                      const h3_st_tensor *tensor,
                                      h3_gpu *gpu, const char *name,
                                      size_t elements,
                                      char *error, size_t error_size) {
    _Float16 *raw = malloc(sizeof(_Float16) * elements);
    float *wide = malloc(sizeof(float) * elements);
    h3_gpu_tensor *result = NULL;
    char detail[384];
    if (raw && wide &&
        h3_st_read_data(header, tensor, raw, sizeof(_Float16) * elements,
                        detail, sizeof(detail))) {
        for (size_t index = 0; index < elements; index++)
            wide[index] = (float)raw[index];
        result = h3_gpu_tensor_from_f32(gpu, wide, elements);
        if (!result)
            fail(error, error_size, "cannot upload %s: %s", name,
                 h3_gpu_error(gpu));
    } else {
        fail(error, error_size, "%s",
             raw && wide ? detail : "out of memory widening an F16 weight");
    }
    free(raw);
    free(wide);
    return result;
}

static h3_gpu_tensor *load_tensor(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, h3_dtype dtype,
                                  char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    int converts = (dtype == H3_DTYPE_BF16 &&
                    (tensor->dtype == H3_DTYPE_F16 ||
                     (tensor->dtype == H3_DTYPE_I8 && ndim == 2))) ||
                   (dtype == H3_DTYPE_F32 && tensor->dtype == H3_DTYPE_F16);
    if ((tensor->dtype != dtype && !converts) || tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, h3_dtype_name(tensor->dtype), tensor->ndim,
             h3_dtype_name(dtype), ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    if (tensor->dtype == H3_DTYPE_I8)
        return load_int8_as_bf16(store, gpu, name, shape[0], shape[1],
                                 error, error_size);
    if (tensor->dtype == H3_DTYPE_F16)
        return dtype == H3_DTYPE_BF16 ?
            load_f16_as_bf16(header, tensor, gpu, name, (size_t)elements,
                             error, error_size) :
            load_f16_as_f32(header, tensor, gpu, name, (size_t)elements,
                            error, error_size);
    h3_gpu_tensor *result = dtype == H3_DTYPE_BF16 ?
        h3_gpu_tensor_load_bf16(gpu, header->path, tensor->file_offset,
                                (size_t)elements) :
        h3_gpu_tensor_load_f32(gpu, header->path, tensor->file_offset,
                               (size_t)elements);
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name, h3_gpu_error(gpu));
    }
    return result;
}

h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_BF16,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F32,
                       error, error_size);
}
