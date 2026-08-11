#import "h3_ane_linear.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>
#include <string.h>
#include <time.h>

#define H3_ANE_MAX_CHUNKS 8

struct h3_ane_linear {
    uint32_t input_dim;
    uint32_t padded_dim;
    uint32_t output_dim;
    uint32_t rows;
    uint32_t chunk_dim;
    uint32_t chunks;
    size_t input_bytes;
    size_t output_bytes;
    uint64_t weight_bytes;
    double compile_seconds;
    IOSurfaceRef input_surface[H3_ANE_MAX_CHUNKS];
    IOSurfaceRef output_surface;
    float *input_base[H3_ANE_MAX_CHUNKS];
    float *output_base;
    void *model;
    void *request;
    char *temporary_directory;
};

static void ane_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double ane_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

int h3_ane_linear_available(void) {
    static int state = -1;
    if (state >= 0) return state;
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
           "AppleNeuralEngine", RTLD_NOW);
    state = NSClassFromString(@"_ANEInMemoryModelDescriptor") != nil &&
            NSClassFromString(@"_ANEInMemoryModel") != nil &&
            NSClassFromString(@"_ANERequest") != nil &&
            NSClassFromString(@"_ANEIOSurfaceObject") != nil;
    return state;
}

static NSString *ane_program(uint32_t chunk_dim, uint32_t output_dim,
                             uint32_t chunks, uint32_t rows) {
    NSMutableString *text = [NSMutableString string];
    [text appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\""
        "coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", "
        "\"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n    func main<ios18>("];
    for (uint32_t i = 0; i < chunks; i++)
        [text appendFormat:@"tensor<fp32, [1, %u, 1, %u]> x%u%s",
            chunk_dim, rows, i, i + 1 == chunks ? "" : ", "];
    [text appendString:@") {\n"
        "            string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n"
        "            tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n"
        "            tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "            tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n"
        "            int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n"
        "            string f16 = const()[name = string(\"f16\"), val = string(\"fp16\")];\n"];
    unsigned long stride = 64ul + (unsigned long)chunk_dim * output_dim * 2ul;
    for (uint32_t i = 0; i < chunks; i++) {
        [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> h%u = "
            "cast(dtype = f16, x = x%u)[name = string(\"ci%u\")];\n",
            chunk_dim, rows, i, i, i];
        [text appendFormat:@"            tensor<fp16, [%u, %u, 1, 1]> W%u = "
            "const()[name = string(\"W%u\"), val = tensor<fp16, [%u, %u, 1, 1]>("
            "BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), "
            "offset = uint64(%lu)))];\n",
            output_dim, chunk_dim, i, i, output_dim, chunk_dim,
            64ul + (unsigned long)i * stride];
        [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> p%u = "
            "conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, "
            "strides = st, weight = W%u, x = h%u)[name = string(\"p%u\")];\n",
            output_dim, rows, i, i, i, i];
    }
    NSString *previous = @"p0";
    for (uint32_t i = 1; i < chunks; i++) {
        NSString *sum = [NSString stringWithFormat:@"s%u", i];
        [text appendFormat:@"            tensor<fp16, [1, %u, 1, %u]> %@ = "
            "add(x = %@, y = p%u)[name = string(\"%@\")];\n",
            output_dim, rows, sum, previous, i, sum];
        previous = sum;
    }
    [text appendString:@"            string f32 = const()[name = string(\"f32\"), "
        "val = string(\"fp32\")];\n"];
    [text appendFormat:@"            tensor<fp32, [1, %u, 1, %u]> c = "
        "cast(dtype = f32, x = %@)[name = string(\"co\")];\n",
        output_dim, rows, previous];
    [text appendString:@"        } -> (c);\n}\n"];
    return text;
}

/* 64-byte file header, then one 64-byte chunk header per weight tile. The
 * chunk header carries the payload size at +8 and the payload offset from the
 * start of the file at +16. */
static uint8_t *ane_weight_blob(const void *weights, h3_ane_weight_dtype dtype,
                                uint32_t input_dim, uint32_t output_dim,
                                uint32_t chunk_dim, uint32_t chunks,
                                size_t *out_bytes) {
    size_t tile = (size_t)output_dim * chunk_dim * 2;
    size_t stride = 64 + tile;
    size_t total = 64 + stride * chunks;
    uint8_t *blob = calloc(total, 1);
    if (!blob) return NULL;
    blob[0] = 0x01;
    blob[4] = 0x02;
    for (uint32_t c = 0; c < chunks; c++) {
        uint8_t *header = blob + 64 + (size_t)c * stride;
        header[0] = 0xEF; header[1] = 0xBE; header[2] = 0xAD; header[3] = 0xDE;
        header[4] = 0x01;
        uint32_t payload = (uint32_t)tile;
        uint32_t offset = (uint32_t)(64 + (size_t)c * stride + 64);
        memcpy(header + 8, &payload, sizeof(payload));
        memcpy(header + 16, &offset, sizeof(offset));
        __fp16 *destination = (__fp16 *)(void *)(header + 64);
        uint32_t base = c * chunk_dim;
        uint32_t span = input_dim > base ?
            (input_dim - base < chunk_dim ? input_dim - base : chunk_dim) : 0;
        for (uint32_t n = 0; n < output_dim; n++) {
            const uint16_t *source =
                (const uint16_t *)weights + (size_t)n * input_dim + base;
            __fp16 *row = destination + (size_t)n * chunk_dim;
            if (dtype == H3_ANE_W_F16) {
                memcpy(row, source, (size_t)span * 2);
            } else {
                for (uint32_t k = 0; k < span; k++) {
                    uint32_t wide = (uint32_t)source[k] << 16;
                    float value;
                    memcpy(&value, &wide, sizeof(value));
                    row[k] = (__fp16)value;
                }
            }
        }
    }
    *out_bytes = total;
    return blob;
}

static uint8_t *ane_weight_blob_chunked(const void *chunked, size_t bytes,
                                        uint32_t output_dim,
                                        uint32_t chunk_dim, uint32_t chunks,
                                        size_t *out_bytes) {
    size_t tile = (size_t)output_dim * chunk_dim * 2;
    if (bytes != tile * chunks) return NULL;
    size_t stride = 64 + tile;
    size_t total = 64 + stride * chunks;
    uint8_t *blob = calloc(total, 1);
    if (!blob) return NULL;
    blob[0] = 0x01;
    blob[4] = 0x02;
    for (uint32_t c = 0; c < chunks; c++) {
        uint8_t *header = blob + 64 + (size_t)c * stride;
        header[0] = 0xEF; header[1] = 0xBE; header[2] = 0xAD; header[3] = 0xDE;
        header[4] = 0x01;
        uint32_t payload = (uint32_t)tile;
        uint32_t offset = (uint32_t)(64 + (size_t)c * stride + 64);
        memcpy(header + 8, &payload, sizeof(payload));
        memcpy(header + 16, &offset, sizeof(offset));
        memcpy(header + 64, (const uint8_t *)chunked + (size_t)c * tile, tile);
    }
    *out_bytes = total;
    return blob;
}

static IOSurfaceRef ane_surface(size_t bytes) {
    size_t aligned = (bytes + 16383u) & ~(size_t)16383u;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(aligned),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(aligned),
        (id)kIOSurfaceAllocSize: @(aligned),
        (id)kIOSurfacePixelFormat: @0});
}

static h3_ane_linear *ane_build(const char *name, uint8_t *blob,
                                size_t blob_bytes, uint32_t input_dim,
                                uint32_t output_dim, uint32_t rows,
                                uint32_t chunk_dim, uint32_t chunks,
                                char *error, size_t error_size) {
    h3_ane_linear *linear = calloc(1, sizeof(*linear));
    if (!linear) {
        free(blob);
        ane_fail(error, error_size, "out of memory creating ANE %s", name);
        return NULL;
    }
    linear->input_dim = input_dim;
    linear->padded_dim = chunk_dim * chunks;
    linear->output_dim = output_dim;
    linear->rows = rows;
    linear->chunk_dim = chunk_dim;
    linear->chunks = chunks;
    linear->input_bytes = (size_t)chunk_dim * rows * sizeof(float);
    linear->output_bytes = (size_t)output_dim * rows * sizeof(float);
    linear->weight_bytes = blob_bytes;

    @autoreleasepool {
        NSError *failure = nil;
        NSData *weights = [NSData dataWithBytesNoCopy:blob length:blob_bytes
                                         freeWhenDone:YES];
        NSData *program = [ane_program(chunk_dim, output_dim, chunks, rows)
            dataUsingEncoding:NSUTF8StringEncoding];
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
            program, @{@"@model_path/weights/weight.bin":
                       @{@"offset": @0, @"data": weights}}, nil);
        if (!descriptor) {
            ane_fail(error, error_size, "ANE %s descriptor rejected", name);
            free(linear);
            return NULL;
        }
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
        if (!model) {
            ane_fail(error, error_size, "ANE %s model rejected", name);
            free(linear);
            return NULL;
        }
        NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(
            model, @selector(hexStringIdentifier));
        NSString *directory = [NSTemporaryDirectory()
            stringByAppendingPathComponent:identifier];
        NSFileManager *files = [NSFileManager defaultManager];
        [files createDirectoryAtPath:
            [directory stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [program writeToFile:[directory stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];
        [weights writeToFile:[directory
            stringByAppendingPathComponent:@"weights/weight.bin"]
                  atomically:YES];
        linear->temporary_directory = strdup(directory.UTF8String);
        double started = ane_seconds();
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(compileWithQoS:options:error:), 21, @{},
                &failure)) {
            ane_fail(error, error_size, "ANE %s compile failed: %s", name,
                     failure ? failure.localizedDescription.UTF8String : "?");
            [files removeItemAtPath:directory error:nil];
            free(linear->temporary_directory);
            free(linear);
            return NULL;
        }
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{},
                &failure)) {
            ane_fail(error, error_size, "ANE %s load failed: %s", name,
                     failure ? failure.localizedDescription.UTF8String : "?");
            [files removeItemAtPath:directory error:nil];
            free(linear->temporary_directory);
            free(linear);
            return NULL;
        }
        linear->compile_seconds = ane_seconds() - started;
        NSMutableArray *inputs = [NSMutableArray array];
        NSMutableArray *indices = [NSMutableArray array];
        for (uint32_t c = 0; c < chunks; c++) {
            linear->input_surface[c] = ane_surface(linear->input_bytes);
            if (!linear->input_surface[c]) {
                ane_fail(error, error_size, "ANE %s input surface failed", name);
                free(linear->temporary_directory);
                free(linear);
                return NULL;
            }
            linear->input_base[c] =
                IOSurfaceGetBaseAddress(linear->input_surface[c]);
            memset(linear->input_base[c], 0, linear->input_bytes);
            [inputs addObject:((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:),
                linear->input_surface[c])];
            [indices addObject:@(c)];
        }
        linear->output_surface = ane_surface(linear->output_bytes);
        if (!linear->output_surface) {
            ane_fail(error, error_size, "ANE %s output surface failed", name);
            free(linear->temporary_directory);
            free(linear);
            return NULL;
        }
        linear->output_base = IOSurfaceGetBaseAddress(linear->output_surface);
        id output = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:),
            linear->output_surface);
        id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            requestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                      weightsBuffer:perfStats:procedureIndex:),
            inputs, indices, @[output], @[@0], nil, nil, @0);
        if (!request) {
            ane_fail(error, error_size, "ANE %s request rejected", name);
            free(linear->temporary_directory);
            free(linear);
            return NULL;
        }
        linear->model = (__bridge_retained void *)model;
        linear->request = (__bridge_retained void *)request;
    }
    return linear;
}

static int ane_shape(const char *name, uint32_t input_dim, uint32_t output_dim,
                     uint32_t rows, uint32_t kc, uint32_t *chunk_dim,
                     uint32_t *chunks, char *error, size_t error_size) {
    if (!input_dim || !output_dim || !rows || !kc) {
        ane_fail(error, error_size, "ANE %s has an empty dimension", name);
        return 0;
    }
    if (!h3_ane_linear_available()) {
        ane_fail(error, error_size, "the Neural Engine bridge is unavailable");
        return 0;
    }
    uint32_t count = (input_dim + kc - 1) / kc;
    if (count > H3_ANE_MAX_CHUNKS) {
        ane_fail(error, error_size,
                 "ANE %s needs %u inputs; request binding is only reliable "
                 "through %d", name, count, H3_ANE_MAX_CHUNKS);
        return 0;
    }
    *chunk_dim = kc;
    *chunks = count;
    return 1;
}

h3_ane_linear *h3_ane_linear_create(const char *name, const void *weights,
                                    h3_ane_weight_dtype dtype,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size) {
    uint32_t chunk_dim = 0, chunks = 0;
    if (!weights || !ane_shape(name, input_dim, output_dim, rows, kc,
                               &chunk_dim, &chunks, error, error_size))
        return NULL;
    size_t blob_bytes = 0;
    uint8_t *blob = ane_weight_blob(weights, dtype, input_dim, output_dim,
                                    chunk_dim, chunks, &blob_bytes);
    if (!blob) {
        ane_fail(error, error_size, "out of memory building ANE %s weights",
                 name);
        return NULL;
    }
    return ane_build(name, blob, blob_bytes, input_dim, output_dim, rows,
                     chunk_dim, chunks, error, error_size);
}

h3_ane_linear *h3_ane_linear_create_chunked(const char *name,
                                    const void *chunks_data, size_t chunk_bytes,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size) {
    uint32_t chunk_dim = 0, chunks = 0;
    if (!chunks_data || !ane_shape(name, input_dim, output_dim, rows, kc,
                                   &chunk_dim, &chunks, error, error_size))
        return NULL;
    size_t blob_bytes = 0;
    uint8_t *blob = ane_weight_blob_chunked(chunks_data, chunk_bytes,
                                            output_dim, chunk_dim, chunks,
                                            &blob_bytes);
    if (!blob) {
        ane_fail(error, error_size,
                 "ANE %s chunk payload is %zu bytes, expected %zu", name,
                 chunk_bytes, (size_t)output_dim * chunk_dim * 2 * chunks);
        return NULL;
    }
    return ane_build(name, blob, blob_bytes, input_dim, output_dim, rows,
                     chunk_dim, chunks, error, error_size);
}

void h3_ane_linear_free(h3_ane_linear *linear) {
    if (!linear) return;
    @autoreleasepool {
        if (linear->model) {
            id model = (__bridge_transfer id)linear->model;
            NSError *failure = nil;
            ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &failure);
            linear->model = NULL;
        }
        if (linear->request) {
            id request = (__bridge_transfer id)linear->request;
            (void)request;
            linear->request = NULL;
        }
        if (linear->temporary_directory) {
            [[NSFileManager defaultManager]
                removeItemAtPath:@(linear->temporary_directory) error:nil];
            free(linear->temporary_directory);
        }
    }
    for (uint32_t c = 0; c < linear->chunks; c++)
        if (linear->input_surface[c]) CFRelease(linear->input_surface[c]);
    if (linear->output_surface) CFRelease(linear->output_surface);
    free(linear);
}

uint32_t h3_ane_linear_chunks(const h3_ane_linear *linear) {
    return linear ? linear->chunks : 0;
}

uint32_t h3_ane_linear_chunk_dim(const h3_ane_linear *linear) {
    return linear ? linear->chunk_dim : 0;
}

uint32_t h3_ane_linear_rows(const h3_ane_linear *linear) {
    return linear ? linear->rows : 0;
}

uint32_t h3_ane_linear_output_dim(const h3_ane_linear *linear) {
    return linear ? linear->output_dim : 0;
}

float *h3_ane_linear_input(h3_ane_linear *linear, uint32_t chunk) {
    if (!linear || chunk >= linear->chunks) return NULL;
    return linear->input_base[chunk];
}

float *h3_ane_linear_output(h3_ane_linear *linear) {
    return linear ? linear->output_base : NULL;
}

size_t h3_ane_linear_input_bytes(const h3_ane_linear *linear) {
    return linear ? linear->input_bytes : 0;
}

size_t h3_ane_linear_output_bytes(const h3_ane_linear *linear) {
    return linear ? linear->output_bytes : 0;
}

double h3_ane_linear_compile_seconds(const h3_ane_linear *linear) {
    return linear ? linear->compile_seconds : 0.0;
}

uint64_t h3_ane_linear_weight_bytes(const h3_ane_linear *linear) {
    return linear ? linear->weight_bytes : 0;
}

int h3_ane_linear_eval(h3_ane_linear *linear, char *error, size_t error_size) {
    if (!linear || !linear->model || !linear->request) {
        ane_fail(error, error_size, "the ANE projection is not loaded");
        return 0;
    }
    int ok = 0;
    @autoreleasepool {
        NSError *failure = nil;
        id model = (__bridge id)linear->model;
        id request = (__bridge id)linear->request;
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:), 21, @{},
            request, &failure) ? 1 : 0;
        if (!ok)
            ane_fail(error, error_size, "ANE evaluation failed: %s",
                     failure ? failure.localizedDescription.UTF8String : "?");
    }
    return ok;
}

struct h3_ane_projection {
    h3_ane_linear *linear;
    h3_gpu_tensor *plane[H3_ANE_MAX_CHUNKS];
    h3_gpu_tensor *result;
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t rows;
    double pack_seconds;
    double eval_seconds;
    uint64_t calls;
};

uint32_t h3_ane_linear_default_chunk(uint32_t input_dim) {
    uint32_t chunk = 1024;
    while ((input_dim + chunk - 1) / chunk > H3_ANE_MAX_CHUNKS) chunk += 1024;
    return chunk;
}

h3_ane_projection *h3_ane_projection_create(h3_gpu *gpu, const char *name,
                                    const h3_gpu_tensor *weight,
                                    uint32_t input_dim, uint32_t output_dim,
                                    uint32_t rows, uint32_t kc,
                                    char *error, size_t error_size) {
    const void *values = h3_gpu_tensor_host_pointer(weight);
    if (!values) {
        ane_fail(error, error_size, "ANE %s has no host weight", name);
        return NULL;
    }
    if (h3_gpu_tensor_elements(weight) != (size_t)input_dim * output_dim ||
        h3_gpu_tensor_dtype(weight) != H3_GPU_BF16) {
        ane_fail(error, error_size, "ANE %s weight is not [%u][%u] BF16", name,
                 output_dim, input_dim);
        return NULL;
    }
    h3_ane_projection *projection = calloc(1, sizeof(*projection));
    if (!projection) {
        ane_fail(error, error_size, "out of memory creating ANE %s", name);
        return NULL;
    }
    projection->input_dim = input_dim;
    projection->output_dim = output_dim;
    projection->rows = rows;
    projection->linear = h3_ane_linear_create(name, values, H3_ANE_W_BF16,
                                              input_dim, output_dim, rows, kc,
                                              error, error_size);
    if (!projection->linear) {
        free(projection);
        return NULL;
    }
    uint32_t chunks = h3_ane_linear_chunks(projection->linear);
    uint32_t chunk_dim = h3_ane_linear_chunk_dim(projection->linear);
    for (uint32_t c = 0; c < chunks; c++) {
        projection->plane[c] = h3_gpu_tensor_wrap_f32(
            gpu, h3_ane_linear_input(projection->linear, c),
            (size_t)chunk_dim * rows);
        if (!projection->plane[c]) {
            ane_fail(error, error_size, "cannot share ANE %s plane %u", name, c);
            h3_ane_projection_free(projection);
            return NULL;
        }
    }
    projection->result = h3_gpu_tensor_wrap_f32(
        gpu, h3_ane_linear_output(projection->linear),
        (size_t)output_dim * rows);
    if (!projection->result) {
        ane_fail(error, error_size, "cannot share ANE %s result", name);
        h3_ane_projection_free(projection);
        return NULL;
    }
    return projection;
}

void h3_ane_projection_free(h3_ane_projection *projection) {
    if (!projection) return;
    for (uint32_t c = 0; c < H3_ANE_MAX_CHUNKS; c++)
        if (projection->plane[c]) h3_gpu_tensor_free(projection->plane[c]);
    if (projection->result) h3_gpu_tensor_free(projection->result);
    h3_ane_linear_free(projection->linear);
    free(projection);
}

int h3_ane_projection_apply(h3_ane_projection *projection, h3_gpu *gpu,
                            h3_gpu_tensor *output, const h3_gpu_tensor *input,
                            char *error, size_t error_size) {
    if (!projection || !gpu) {
        ane_fail(error, error_size, "the ANE projection is missing");
        return 0;
    }
    uint32_t chunks = h3_ane_linear_chunks(projection->linear);
    uint32_t chunk_dim = h3_ane_linear_chunk_dim(projection->linear);
    double started = ane_seconds();
    for (uint32_t c = 0; c < chunks; c++)
        if (!h3_gpu_pack_ane_input_bf16(gpu, projection->plane[c], input,
                                        projection->rows, projection->input_dim,
                                        c * chunk_dim, chunk_dim)) {
            ane_fail(error, error_size, "ANE pack failed: %s",
                     h3_gpu_error(gpu));
            return 0;
        }
    if (!h3_gpu_submit(gpu)) {
        ane_fail(error, error_size, "ANE pack submit failed: %s",
                 h3_gpu_error(gpu));
        return 0;
    }
    double packed = ane_seconds();
    if (!h3_ane_linear_eval(projection->linear, error, error_size)) return 0;
    double evaluated = ane_seconds();
    if (!h3_gpu_begin(gpu)) {
        ane_fail(error, error_size, "cannot resume after the ANE: %s",
                 h3_gpu_error(gpu));
        return 0;
    }
    if (!h3_gpu_unpack_ane_output_bf16(gpu, output, projection->result,
                                       projection->rows,
                                       projection->output_dim)) {
        ane_fail(error, error_size, "ANE unpack failed: %s", h3_gpu_error(gpu));
        return 0;
    }
    projection->pack_seconds += packed - started;
    projection->eval_seconds += evaluated - packed;
    projection->calls++;
    return 1;
}

void h3_ane_projection_timings(const h3_ane_projection *projection,
                               double *pack_seconds, double *eval_seconds,
                               uint64_t *calls) {
    if (pack_seconds) *pack_seconds = projection ? projection->pack_seconds : 0.0;
    if (eval_seconds) *eval_seconds = projection ? projection->eval_seconds : 0.0;
    if (calls) *calls = projection ? projection->calls : 0;
}

uint64_t h3_ane_projection_weight_bytes(const h3_ane_projection *projection) {
    return projection ? h3_ane_linear_weight_bytes(projection->linear) : 0;
}

double h3_ane_projection_compile_seconds(const h3_ane_projection *projection) {
    return projection ? h3_ane_linear_compile_seconds(projection->linear) : 0.0;
}
