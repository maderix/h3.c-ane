#import "h3_ane_block.h"
#import "h3_convrot.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#import <objc/runtime.h>

#include <dlfcn.h>
#include <math.h>
#include <string.h>
#include <time.h>

enum {
    BLK_HIDDEN = 5376,
    BLK_HEADS = 56,
    BLK_HEAD_DIM = 128,
    BLK_INNER = BLK_HEADS * BLK_HEAD_DIM,
    BLK_FFN = 14336,
    BLK_ROPE_HALF = 48,
    BLK_ROPE_DIMS = BLK_ROPE_HALF * 2,
    BLK_GROUP = 256,
    BLK_ROW_MULTIPLE = 16
};

struct h3_ane_block {
    uint32_t rows;
    uint32_t padded_rows;
    uint64_t weight_bytes;
    double compile_seconds;
    IOSurfaceRef input_surface;
    IOSurfaceRef mod_surface;
    IOSurfaceRef output_surface;
    float *input_base;
    float *mod_base;
    float *output_base;
    void *model;
    void *request;
    char *temporary_directory;
};

static void blk_fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static double blk_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

/* Blob assembly: 64-byte file header, then 64-byte-aligned chunks. */
typedef struct {
    uint8_t *data;
    size_t length;
    size_t cursor;
} blk_blob;

static size_t blk_blob_add(blk_blob *blob, const void *payload, size_t bytes) {
    size_t header = blob->cursor;
    uint8_t *chunk = blob->data + header;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    uint32_t size32 = (uint32_t)bytes;
    uint32_t offset32 = (uint32_t)(header + 64);
    memcpy(chunk + 8, &size32, sizeof(size32));
    memcpy(chunk + 16, &offset32, sizeof(offset32));
    if (payload) memcpy(chunk + 64, payload, bytes);
    blob->cursor = header + 64 + ((bytes + 63) & ~(size_t)63);
    return header;
}

static __fp16 *blk_blob_reserve(blk_blob *blob, size_t bytes, size_t *header) {
    *header = blk_blob_add(blob, NULL, bytes);
    return (__fp16 *)(void *)(blob->data + *header + 64);
}

typedef struct {
    size_t weight;   /* int8 rows */
    size_t scales;   /* fp16 [rows] */
    size_t rotation; /* fp16 [k][group] */
    uint32_t rows;
    uint32_t k;
} blk_projection;

static int blk_pack_projection(blk_blob *blob, const h3_weight_store *store,
                               const char *prefix, const char *stem,
                               uint32_t rows, uint32_t k, size_t rotation,
                               blk_projection *projection,
                               char *error, size_t error_size) {
    char name[192];
    snprintf(name, sizeof(name), "%s%s.weight", prefix, stem);
    int8_t *quantized = NULL;
    float *scales = NULL;
    int group_size = 0;
    if (!h3_weight_load_int8_raw(store, name, rows, k, &quantized, &scales,
                                 &group_size, error, error_size)) return 0;
    if (group_size != BLK_GROUP) {
        blk_fail(error, error_size, "%s has convrot group %d, expected %d",
                 name, group_size, BLK_GROUP);
        free(quantized); free(scales);
        return 0;
    }
    projection->weight = blk_blob_add(blob, quantized, (size_t)rows * k);
    size_t header;
    __fp16 *packed = blk_blob_reserve(blob, (size_t)rows * 2, &header);
    for (uint32_t row = 0; row < rows; row++) packed[row] = (__fp16)scales[row];
    projection->scales = header;
    projection->rotation = rotation;
    projection->rows = rows;
    projection->k = k;
    free(quantized);
    free(scales);
    return 1;
}

static size_t blk_pack_rotation(blk_blob *blob, uint32_t k) {
    const float *table = h3_convrot_hadamard(BLK_GROUP);
    size_t header;
    __fp16 *packed = blk_blob_reserve(blob, (size_t)k * BLK_GROUP * 2, &header);
    for (uint32_t o = 0; o < k; o++)
        for (uint32_t j = 0; j < BLK_GROUP; j++)
            packed[(size_t)o * BLK_GROUP + j] =
                (__fp16)table[j * BLK_GROUP + o % BLK_GROUP];
    return header;
}

static int blk_pack_norm(blk_blob *blob, const h3_weight_store *store,
                         const char *prefix, const char *stem, uint32_t width,
                         size_t *header_out, char *error, size_t error_size) {
    char name[192];
    snprintf(name, sizeof(name), "%s%s", prefix, stem);
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor || tensor->dtype != H3_DTYPE_BF16 ||
        h3_st_tensor_elements(tensor) != width) {
        blk_fail(error, error_size, "norm weight %s is not BF16[%u]", name,
                 width);
        return 0;
    }
    uint16_t *raw = malloc(width * sizeof(*raw));
    char detail[384];
    if (!raw || !h3_st_read_data(header, tensor, raw, width * sizeof(*raw),
                                 detail, sizeof(detail))) {
        blk_fail(error, error_size, "%s", raw ? detail : "out of memory");
        free(raw);
        return 0;
    }
    size_t chunk;
    __fp16 *packed = blk_blob_reserve(blob, (size_t)width * 2, &chunk);
    for (uint32_t index = 0; index < width; index++) {
        uint32_t bits = (uint32_t)raw[index] << 16;
        float value;
        memcpy(&value, &bits, sizeof(value));
        packed[index] = (__fp16)value;
    }
    free(raw);
    *header_out = chunk;
    return 1;
}

/* --- MIL emission helpers ------------------------------------------------ */

#define EMIT(...) [text appendFormat:__VA_ARGS__]

static void emit_const_blob(NSMutableString *text, const char *name,
                            const char *dtype, const char *shape,
                            size_t header) {
    EMIT(@"        tensor<%s, %s> %s = const()[name=string(\"%s\"), "
          "val=tensor<%s, %s>(BLOBFILE(path=string(\"@model_path/weights/"
          "weight.bin\"), offset=uint64(%zu)))];\n",
         dtype, shape, name, name, dtype, shape, header);
}

/* Fp16 RMSNorm over the channel axis with a broadcast gamma. */
static void emit_rms_norm(NSMutableString *text, const char *tag,
                          const char *source, const char *gamma,
                          uint32_t channels, uint32_t rows) {
    /* pre-scale keeps x^2 inside fp16 for the grown late-block residuals */
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s_p = mul(x=%s, y=inv256)"
          "[name=string(\"%s_p\")];\n", channels, rows, tag, source, tag);
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s_sq = mul(x=%s_p, y=%s_p)"
          "[name=string(\"%s_sq\")];\n", channels, rows, tag, tag, tag,
         tag);
    EMIT(@"        tensor<fp16, [1,1,1,%u]> %s_s = reduce_sum(axes=ch_axis, "
          "keep_dims=keep, x=%s_sq)[name=string(\"%s_s\")];\n",
         rows, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,1,1,%u]> %s_m = mul(x=%s_s, y=inv_hidden)"
          "[name=string(\"%s_m\")];\n", rows, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,1,1,%u]> %s_e = add(x=%s_m, y=norm_eps)"
          "[name=string(\"%s_e\")];\n", rows, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,1,1,%u]> %s_r = pow(x=%s_e, y=nhalf)"
          "[name=string(\"%s_r\")];\n", rows, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s_n = mul(x=%s_p, y=%s_r)"
          "[name=string(\"%s_n\")];\n", channels, rows, tag, tag, tag,
         tag);
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s = mul(x=%s_n, y=%s)"
          "[name=string(\"%s\")];\n", channels, rows, tag, tag, gamma, tag);
}

/* Per-segment y = x * (1 + scale[slot]) + shift[slot], slots from the packed
 * mod tensor; then concatenated back in sequence order. */
static void emit_mod_scale_shift(NSMutableString *text, const char *tag,
                                 const char *source, int slot_shift,
                                 int slot_scale, uint32_t channels,
                                 const uint32_t *ends, uint32_t segments,
                                 uint32_t rows) {
    uint32_t begin = 0;
    NSMutableString *parts = [NSMutableString string];
    for (uint32_t seg = 0; seg < segments; seg++) {
        uint32_t width = ends[seg] - begin;
        EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s_x%u = "
              "slice_by_size(x=%s, begin=seq_b%u, size=seq_s%u_%u)"
              "[name=string(\"%s_x%u\")];\n",
             channels, width, tag, seg, source, seg, seg, width, tag, seg);
        EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s_sc%u = mul(x=%s_x%u, "
              "y=mod_%u_%d)[name=string(\"%s_sc%u\")];\n",
             channels, width, tag, seg, tag, seg, seg, slot_scale, tag, seg);
        EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s_a%u = add(x=%s_sc%u, "
              "y=%s_x%u)[name=string(\"%s_a%u\")];\n",
             channels, width, tag, seg, tag, seg, tag, seg, tag, seg);
        EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s_y%u = add(x=%s_a%u, "
              "y=mod_%u_%d)[name=string(\"%s_y%u\")];\n",
             channels, width, tag, seg, tag, seg, seg, slot_shift, tag, seg);
        [parts appendFormat:@"%s%s_y%u", seg ? "," : "", tag, seg];
        begin = ends[seg];
    }
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s = concat(axis=seq_axis, "
          "interleave=noil, values=(%@))[name=string(\"%s\")];\n",
         channels, rows, tag, parts, tag);
}

/* Per-segment x += other * gate[slot]. */
static void emit_mod_gate(NSMutableString *text, const char *tag,
                          const char *carry, const char *other, int slot_gate,
                          uint32_t channels, const uint32_t *ends,
                          uint32_t segments, uint32_t rows) {
    uint32_t begin = 0;
    NSMutableString *parts = [NSMutableString string];
    for (uint32_t seg = 0; seg < segments; seg++) {
        uint32_t width = ends[seg] - begin;
        EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s_o%u = "
              "slice_by_size(x=%s, begin=seq_b%u, size=seq_s%u_%u)"
              "[name=string(\"%s_o%u\")];\n",
             channels, width, tag, seg, other, seg, seg, width, tag, seg);
        EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s_c%u = "
              "slice_by_size(x=%s, begin=seq_b%u, size=seq_s%u_%u)"
              "[name=string(\"%s_c%u\")];\n",
             channels, width, tag, seg, carry, seg, seg, width, tag, seg);
        EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s_g%u = mul(x=%s_o%u, "
              "y=mod_%u_%d)[name=string(\"%s_g%u\")];\n",
             channels, width, tag, seg, tag, seg, seg, slot_gate, tag, seg);
        EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s_y%u = add(x=%s_c%u, "
              "y=%s_g%u)[name=string(\"%s_y%u\")];\n",
             channels, width, tag, seg, tag, seg, tag, seg, tag, seg);
        [parts appendFormat:@"%s%s_y%u", seg ? "," : "", tag, seg];
        begin = ends[seg];
    }
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s = concat(axis=seq_axis, "
          "interleave=noil, values=(%@))[name=string(\"%s\")];\n",
         channels, rows, tag, parts, tag);
}

/* Grouped Hadamard rotation followed by the int8 projection. */
static void emit_projection(NSMutableString *text, const char *tag,
                            const char *source, const blk_projection *p,
                            uint32_t rows) {
    EMIT(@"        int32 %s_gr = const()[name=string(\"%s_gr\"), "
          "val=int32(%u)];\n", tag, tag, p->k / BLK_GROUP);
    char hr[64], w[64];
    snprintf(hr, sizeof(hr), "%s_hr", tag);
    snprintf(w, sizeof(w), "%s_wt", tag);
    char shape[96];
    snprintf(shape, sizeof(shape), "[%u,%u,1,1]", p->k, BLK_GROUP);
    emit_const_blob(text, hr, "fp16", shape, p->rotation);
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s_r = conv(dilations=dl, "
          "groups=%s_gr, pad=pd, pad_type=pt, strides=st, weight=%s, x=%s)"
          "[name=string(\"%s_r\")];\n",
         p->k, rows, tag, tag, hr, source, tag);
    EMIT(@"        tensor<fp16, [%u,%u,1,1]> %s = "
          "constexpr_affine_dequantize()[axis=int32(0), name=string(\"%s\"), "
          "quantized_data=tensor<int8, [%u,%u,1,1]>(BLOBFILE("
          "path=string(\"@model_path/weights/weight.bin\"), "
          "offset=uint64(%zu))), scale=tensor<fp16, [%u]>(BLOBFILE("
          "path=string(\"@model_path/weights/weight.bin\"), "
          "offset=uint64(%zu))), zero_point=int8(0)];\n",
         p->rows, p->k, w, w, p->rows, p->k, p->weight, p->rows, p->scales);
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s = conv(dilations=dl, "
          "groups=g1, pad=pd, pad_type=pt, strides=st, weight=%s, x=%s_r)"
          "[name=string(\"%s\")];\n",
         p->rows, rows, tag, w, tag, tag);
}

/* Per-head fp32 RMSNorm + gamma + 48-pair RoPE for one [1,H,S,D] tensor. */
static void emit_head_norm_rope(NSMutableString *text, const char *tag,
                                const char *source, const char *gamma,
                                uint32_t rows) {
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_sq = mul(x=%s, y=%s)"
          "[name=string(\"%s_sq\")];\n", BLK_HEADS, rows, BLK_HEAD_DIM, tag,
         source, source, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,1]> %s_ss = reduce_sum(axes=hd_axis, "
          "keep_dims=keep, x=%s_sq)[name=string(\"%s_ss\")];\n",
         BLK_HEADS, rows, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,1]> %s_m = mul(x=%s_ss, y=inv_head)"
          "[name=string(\"%s_m\")];\n", BLK_HEADS, rows, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,1]> %s_e = add(x=%s_m, y=norm_eps)"
          "[name=string(\"%s_e\")];\n", BLK_HEADS, rows, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,1]> %s_rs = pow(x=%s_e, y=nhalf)"
          "[name=string(\"%s_rs\")];\n", BLK_HEADS, rows, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_n = mul(x=%s, y=%s_rs)"
          "[name=string(\"%s_n\")];\n", BLK_HEADS, rows, BLK_HEAD_DIM, tag,
         source, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_g = mul(x=%s_n, y=%s)"
          "[name=string(\"%s_g\")];\n", BLK_HEADS, rows, BLK_HEAD_DIM, tag,
         tag, gamma, tag);
    /* dims [0,96) rotate in 48-pairs, [96,128) pass through */
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_rot = slice_by_size(x=%s_g, "
          "begin=rope_b0, size=rope_s96)[name=string(\"%s_rot\")];\n",
         BLK_HEADS, rows, BLK_ROPE_DIMS, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_tail = slice_by_size(x=%s_g, "
          "begin=rope_b96, size=rope_s32)[name=string(\"%s_tail\")];\n",
         BLK_HEADS, rows, BLK_HEAD_DIM - BLK_ROPE_DIMS, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_lo = slice_by_size(x=%s_rot, "
          "begin=rope_b0, size=rope_s48)[name=string(\"%s_lo\")];\n",
         BLK_HEADS, rows, BLK_ROPE_HALF, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_hi = slice_by_size(x=%s_rot, "
          "begin=rope_b48, size=rope_s48)[name=string(\"%s_hi\")];\n",
         BLK_HEADS, rows, BLK_ROPE_HALF, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_nhi = mul(x=%s_hi, y=neg1)"
          "[name=string(\"%s_nhi\")];\n", BLK_HEADS, rows, BLK_ROPE_HALF, tag,
         tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_half = concat(axis=hd_cat, "
          "interleave=noil, values=(%s_nhi,%s_lo))[name=string(\"%s_half\")];\n",
         BLK_HEADS, rows, BLK_ROPE_DIMS, tag, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_c = mul(x=%s_rot, y=rope_cos)"
          "[name=string(\"%s_c\")];\n", BLK_HEADS, rows, BLK_ROPE_DIMS, tag,
         tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_s = mul(x=%s_half, "
          "y=rope_sin)[name=string(\"%s_s\")];\n", BLK_HEADS, rows,
         BLK_ROPE_DIMS, tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_ro = add(x=%s_c, y=%s_s)"
          "[name=string(\"%s_ro\")];\n", BLK_HEADS, rows, BLK_ROPE_DIMS, tag,
         tag, tag, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s = concat(axis=hd_cat, "
          "interleave=noil, values=(%s_ro,%s_tail))[name=string(\"%s\")];\n",
         BLK_HEADS, rows, BLK_HEAD_DIM, tag, tag, tag, tag);
}

static void emit_heads(NSMutableString *text, const char *tag,
                       const char *source, uint32_t rows) {
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s_4 = reshape(shape=head_sh, "
          "x=%s)[name=string(\"%s_4\")];\n", BLK_HEADS, BLK_HEAD_DIM, rows,
         tag, source, tag);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> %s = transpose(perm=pm, x=%s_4)"
          "[name=string(\"%s\")];\n", BLK_HEADS, rows, BLK_HEAD_DIM, tag, tag,
         tag);
}

#define STAGE_OUT(cond, name, c2, c3, c4) do { \
    if (stage == (cond)) { \
        EMIT(@"        tensor<fp32, [1,%u,%u,%u]> y = cast(dtype=f32t, x=%s)" \
              "[name=string(\"y\")];\n", (uint32_t)(c2), (uint32_t)(c3), \
             (uint32_t)(c4), name); \
        [text appendString:@"        } -> (y);\n}\n"]; \
        return text; \
    } \
} while (0)

static NSString *blk_program(const blk_projection *qkv,
                             const blk_projection *out,
                             const blk_projection *fc1,
                             const blk_projection *fc2,
                             size_t norm1, size_t norm2,
                             size_t q_norm, size_t k_norm,
                             size_t rope_cos, size_t rope_sin,
                             size_t pad_mask, int masked,
                             const uint32_t *ends, uint32_t segments,
                             uint32_t rows) {
    const char *stage_env = getenv("H3_ANE_BLOCK_STAGE");
    int stage = stage_env ? atoi(stage_env) : 99;
    NSMutableString *text = [NSMutableString string];
    char shape1[96];
    [text appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\""
        "coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", "
        "\"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n{\n"];
    EMIT(@"    func main<ios18>(tensor<fp32, [1,%u,1,%u]> i0_x, "
          "tensor<fp32, [1,%u,1,%u]> i1_mods) {\n",
         BLK_HIDDEN, rows, BLK_HIDDEN, (uint32_t)H3_ANE_BLOCK_MOD_WIDTH);
    [text appendString:
        @"        string f16t = const()[name=string(\"f16t\"), val=string(\"fp16\")];\n"
         "        string f32t = const()[name=string(\"f32t\"), val=string(\"fp32\")];\n"];
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> h0 = cast(dtype=f16t, x=i0_x)"
          "[name=string(\"h0\")];\n", BLK_HIDDEN, rows);
    if (stage == 0) {
        EMIT(@"        tensor<fp16, [1,%u,1,%u]> mz = cast(dtype=f16t, x=i1_mods)"
              "[name=string(\"mz\")];\n", BLK_HIDDEN,
             (uint32_t)H3_ANE_BLOCK_MOD_WIDTH);
        [text appendString:
            @"        tensor<int32, [4]> mzb = const()[name=string(\"mzb\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
        EMIT(@"        tensor<int32, [4]> mzs = const()[name=string(\"mzs\"), "
              "val=tensor<int32, [4]>([1,%u,1,1])];\n", BLK_HIDDEN);
        EMIT(@"        tensor<fp16, [1,%u,1,1]> mz0 = slice_by_size(x=mz, "
              "begin=mzb, size=mzs)[name=string(\"mz0\")];\n", BLK_HIDDEN);
        EMIT(@"        tensor<fp16, [1,%u,1,%u]> h0b = add(x=h0, y=mz0)"
              "[name=string(\"h0b\")];\n", BLK_HIDDEN, rows);
    }
    STAGE_OUT(0, "h0b", BLK_HIDDEN, 1, rows);

    /* norms */
    [text appendString:
        @"        tensor<int32, [1]> ch_axis = const()[name=string(\"ch_axis\"), val=tensor<int32, [1]>([1])];\n"
         "        bool keep = const()[name=string(\"keep\"), val=bool(true)];\n"
         "        fp16 norm_eps = const()[name=string(\"norm_eps\"), val=fp16(0.00000006)];\n"
         "        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"
         "        fp16 inv256 = const()[name=string(\"inv256\"), val=fp16(0.00390625)];\n"];
    EMIT(@"        fp16 inv_hidden = const()[name=string(\"inv_hidden\"), "
          "val=fp16(%.10f)];\n", 1.0 / BLK_HIDDEN);
    snprintf(shape1, sizeof(shape1), "[1,%u,1,1]", BLK_HIDDEN);
    emit_const_blob(text, "norm1_g", "fp16", shape1, norm1);
    emit_rms_norm(text, "n1", "h0", "norm1_g", BLK_HIDDEN, rows);
    STAGE_OUT(1, "n1", BLK_HIDDEN, 1, rows);

    /* modulation */
    [text appendString:
        @"        int32 seq_axis = const()[name=string(\"seq_axis\"), val=int32(3)];\n"
         "        bool noil = const()[name=string(\"noil\"), val=bool(false)];\n"];
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> mods_h = cast(dtype=f16t, x=i1_mods)"
          "[name=string(\"mods_h\")];\n", BLK_HIDDEN,
         (uint32_t)H3_ANE_BLOCK_MOD_WIDTH);
    EMIT(@"        tensor<int32, [4]> mod_size = const()[name=string(\"mod_size\"), "
          "val=tensor<int32, [4]>([1,%u,1,1])];\n", BLK_HIDDEN);
    for (uint32_t seg = 0; seg < segments; seg++)
        for (int slot = 0; slot < H3_ANE_BLOCK_SLOTS; slot++) {
            EMIT(@"        tensor<int32, [4]> mod_b%u_%d = const()"
                  "[name=string(\"mod_b%u_%d\"), val=tensor<int32, [4]>"
                  "([0,0,0,%u])];\n", seg, slot, seg, slot,
                 seg * H3_ANE_BLOCK_SLOTS + slot);
            EMIT(@"        tensor<fp16, [1,%u,1,1]> mod_%u_%d = "
                  "slice_by_size(x=mods_h, begin=mod_b%u_%d, size=mod_size)"
                  "[name=string(\"mod_%u_%d\")];\n",
                 BLK_HIDDEN, seg, slot, seg, slot, seg, slot);
        }
    uint32_t begin = 0;
    for (uint32_t seg = 0; seg < segments; seg++) {
        EMIT(@"        tensor<int32, [4]> seq_b%u = const()[name=string(\"seq_b%u\"), "
              "val=tensor<int32, [4]>([0,0,0,%u])];\n", seg, seg, begin);
        uint32_t width = ends[seg] - begin;
        EMIT(@"        tensor<int32, [4]> seq_s%u_%u = const()[name=string(\"seq_s%u_%u\"), "
              "val=tensor<int32, [4]>([1,%u,1,%u])];\n",
             seg, width, seg, width, BLK_HIDDEN, width);
        begin = ends[seg];
    }
    emit_mod_scale_shift(text, "xm1", "n1", 0, 1, BLK_HIDDEN, ends, segments,
                         rows);
    STAGE_OUT(2, "xm1", BLK_HIDDEN, 1, rows);

    /* projections */
    [text appendString:
        @"        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
         "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1, 1])];\n"
         "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n"
         "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1, 1])];\n"
         "        int32 g1 = const()[name=string(\"g1\"), val=int32(1)];\n"];
    emit_projection(text, "qkv", "xm1", qkv, rows);
    STAGE_OUT(3, "qkv", BLK_INNER * 3, 1, rows);

    /* q/k/v slices and head layout */
    [text appendString:
        @"        tensor<int32, [4]> qkv_b0 = const()[name=string(\"qkv_b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    EMIT(@"        tensor<int32, [4]> qkv_b1 = const()[name=string(\"qkv_b1\"), "
          "val=tensor<int32, [4]>([0,%u,0,0])];\n", BLK_INNER);
    EMIT(@"        tensor<int32, [4]> qkv_b2 = const()[name=string(\"qkv_b2\"), "
          "val=tensor<int32, [4]>([0,%u,0,0])];\n", BLK_INNER * 2);
    EMIT(@"        tensor<int32, [4]> qkv_sz = const()[name=string(\"qkv_sz\"), "
          "val=tensor<int32, [4]>([1,%u,1,%u])];\n", BLK_INNER, rows);
    EMIT(@"        tensor<int32, [4]> head_sh = const()[name=string(\"head_sh\"), "
          "val=tensor<int32, [4]>([1,%u,%u,%u])];\n",
         BLK_HEADS, BLK_HEAD_DIM, rows);
    [text appendString:
        @"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0, 1, 3, 2])];\n"];
    const char *names[3] = {"qf", "kf", "vf"};
    const char *begins[3] = {"qkv_b0", "qkv_b1", "qkv_b2"};
    for (int i = 0; i < 3; i++)
        EMIT(@"        tensor<fp16, [1,%u,1,%u]> %s = slice_by_size(x=qkv, "
              "begin=%s, size=qkv_sz)[name=string(\"%s\")];\n",
             BLK_INNER, rows, names[i], begins[i], names[i]);
    emit_heads(text, "qh", "qf", rows);
    emit_heads(text, "kh", "kf", rows);
    emit_heads(text, "vh", "vf", rows);
    STAGE_OUT(4, "qh", BLK_HEADS, rows, BLK_HEAD_DIM);

    /* head norm + rope */
    EMIT(@"        fp16 inv_head = const()[name=string(\"inv_head\"), "
          "val=fp16(%.10f)];\n", 1.0 / BLK_HEAD_DIM);
    [text appendString:
        @"        tensor<int32, [1]> hd_axis = const()[name=string(\"hd_axis\"), val=tensor<int32, [1]>([3])];\n"
         "        int32 hd_cat = const()[name=string(\"hd_cat\"), val=int32(3)];\n"
         "        fp16 neg1 = const()[name=string(\"neg1\"), val=fp16(-1)];\n"
         "        tensor<int32, [4]> rope_b0 = const()[name=string(\"rope_b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    EMIT(@"        tensor<int32, [4]> rope_b48 = const()[name=string(\"rope_b48\"), "
          "val=tensor<int32, [4]>([0,0,0,%u])];\n", BLK_ROPE_HALF);
    EMIT(@"        tensor<int32, [4]> rope_b96 = const()[name=string(\"rope_b96\"), "
          "val=tensor<int32, [4]>([0,0,0,%u])];\n", BLK_ROPE_DIMS);
    EMIT(@"        tensor<int32, [4]> rope_s96 = const()[name=string(\"rope_s96\"), "
          "val=tensor<int32, [4]>([1,%u,%u,%u])];\n",
         BLK_HEADS, rows, BLK_ROPE_DIMS);
    EMIT(@"        tensor<int32, [4]> rope_s48 = const()[name=string(\"rope_s48\"), "
          "val=tensor<int32, [4]>([1,%u,%u,%u])];\n",
         BLK_HEADS, rows, BLK_ROPE_HALF);
    EMIT(@"        tensor<int32, [4]> rope_s32 = const()[name=string(\"rope_s32\"), "
          "val=tensor<int32, [4]>([1,%u,%u,%u])];\n",
         BLK_HEADS, rows, BLK_HEAD_DIM - BLK_ROPE_DIMS);
    snprintf(shape1, sizeof(shape1), "[1,1,1,%u]", BLK_HEAD_DIM);
    emit_const_blob(text, "qn_g", "fp16", shape1, q_norm);
    emit_const_blob(text, "kn_g", "fp16", shape1, k_norm);
    snprintf(shape1, sizeof(shape1), "[1,1,%u,%u]", rows, BLK_ROPE_DIMS);
    emit_const_blob(text, "rope_cos", "fp16", shape1, rope_cos);
    emit_const_blob(text, "rope_sin", "fp16", shape1, rope_sin);
    emit_head_norm_rope(text, "qr", "qh", "qn_g", rows);
    STAGE_OUT(5, "qr", BLK_HEADS, rows, BLK_HEAD_DIM);
    emit_head_norm_rope(text, "kr", "kh", "kn_g", rows);

    /* attention */
    [text appendString:
        @"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"
         "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"
         "        int32 sm_axis = const()[name=string(\"sm_axis\"), val=int32(-1)];\n"];
    EMIT(@"        fp16 att_scale = const()[name=string(\"att_scale\"), "
          "val=fp16(%f)];\n", 1.0 / sqrt((double)BLK_HEAD_DIM));
    EMIT(@"        tensor<int32, [4]> flat_sh = const()[name=string(\"flat_sh\"), "
          "val=tensor<int32, [4]>([1,%u,1,%u])];\n", BLK_INNER, rows);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> sc1 = matmul(transpose_x=bF, "
          "transpose_y=bT, x=qr, y=kr)[name=string(\"sc1\")];\n",
         BLK_HEADS, rows, rows);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> sc2 = mul(x=sc1, y=att_scale)"
          "[name=string(\"sc2\")];\n", BLK_HEADS, rows, rows);
    const char *scored = "sc2";
    if (masked) {
        char mask_shape[64];
        snprintf(mask_shape, sizeof(mask_shape), "[1,1,1,%u]", rows);
        emit_const_blob(text, "pad_mask", "fp16", mask_shape, pad_mask);
        EMIT(@"        tensor<fp16, [1,%u,%u,%u]> sc3 = add(x=sc2, y=pad_mask)"
              "[name=string(\"sc3\")];\n", BLK_HEADS, rows, rows);
        scored = "sc3";
    }
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> probs = softmax(axis=sm_axis, "
          "x=%s)[name=string(\"probs\")];\n", BLK_HEADS, rows, rows, scored);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> att = matmul(transpose_x=bF, "
          "transpose_y=bF, x=probs, y=vh)[name=string(\"att\")];\n",
         BLK_HEADS, rows, BLK_HEAD_DIM);
    EMIT(@"        tensor<fp16, [1,%u,%u,%u]> att_t = transpose(perm=pm, "
          "x=att)[name=string(\"att_t\")];\n", BLK_HEADS, BLK_HEAD_DIM, rows);
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> att_f = reshape(shape=flat_sh, "
          "x=att_t)[name=string(\"att_f\")];\n", BLK_INNER, rows);
    STAGE_OUT(6, "att_f", BLK_INNER, 1, rows);
    emit_projection(text, "aout", "att_f", out, rows);
    STAGE_OUT(7, "aout", BLK_HIDDEN, 1, rows);
    emit_mod_gate(text, "h1", "h0", "aout", 2, BLK_HIDDEN, ends, segments,
                  rows);

    /* MLP */
    snprintf(shape1, sizeof(shape1), "[1,%u,1,1]", BLK_HIDDEN);
    emit_const_blob(text, "norm2_g", "fp16", shape1, norm2);
    emit_rms_norm(text, "n2", "h1", "norm2_g", BLK_HIDDEN, rows);
    emit_mod_scale_shift(text, "xm2", "n2", 3, 4, BLK_HIDDEN, ends, segments,
                         rows);
    STAGE_OUT(8, "xm2", BLK_HIDDEN, 1, rows);
    emit_projection(text, "gu", "xm2", fc1, rows);
    [text appendString:
        @"        tensor<int32, [4]> gu_b0 = const()[name=string(\"gu_b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    EMIT(@"        tensor<int32, [4]> gu_b1 = const()[name=string(\"gu_b1\"), "
          "val=tensor<int32, [4]>([0,%u,0,0])];\n", BLK_FFN);
    EMIT(@"        tensor<int32, [4]> gu_sz = const()[name=string(\"gu_sz\"), "
          "val=tensor<int32, [4]>([1,%u,1,%u])];\n", BLK_FFN, rows);
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> gate = slice_by_size(x=gu, "
          "begin=gu_b0, size=gu_sz)[name=string(\"gate\")];\n", BLK_FFN, rows);
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> up = slice_by_size(x=gu, "
          "begin=gu_b1, size=gu_sz)[name=string(\"up\")];\n", BLK_FFN, rows);
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> gsg = sigmoid(x=gate)"
          "[name=string(\"gsg\")];\n", BLK_FFN, rows);
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> gs = mul(x=gate, y=gsg)"
          "[name=string(\"gs\")];\n", BLK_FFN, rows);
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> act = mul(x=gs, y=up)"
          "[name=string(\"act\")];\n", BLK_FFN, rows);
    STAGE_OUT(9, "act", BLK_FFN, 1, rows);
    /* The trained fc2 output peaks past fp16's 65504 inside the accumulator;
     * a power-of-two pre/post scale keeps every partial sum in range. */
    [text appendString:
        @"        fp16 inv16 = const()[name=string(\"inv16\"), val=fp16(0.0625)];\n"
         "        fp16 sixteen = const()[name=string(\"sixteen\"), val=fp16(16.0)];\n"];
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> act_s = mul(x=act, y=inv16)"
          "[name=string(\"act_s\")];\n", BLK_FFN, rows);
    emit_projection(text, "mout_s", "act_s", fc2, rows);
    EMIT(@"        tensor<fp16, [1,%u,1,%u]> mout = mul(x=mout_s, y=sixteen)"
          "[name=string(\"mout\")];\n", BLK_HIDDEN, rows);
    STAGE_OUT(10, "mout", BLK_HIDDEN, 1, rows);
    emit_mod_gate(text, "h2", "h1", "mout", 5, BLK_HIDDEN, ends, segments,
                  rows);
    EMIT(@"        tensor<fp32, [1,%u,1,%u]> y = cast(dtype=f32t, x=h2)"
          "[name=string(\"y\")];\n", BLK_HIDDEN, rows);
    [text appendString:@"        } -> (y);\n}\n"];
    return text;
}

/* --- runtime ------------------------------------------------------------- */

static IOSurfaceRef blk_surface(size_t bytes) {
    size_t aligned = (bytes + 16383u) & ~(size_t)16383u;
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(aligned),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(aligned),
        (id)kIOSurfaceAllocSize: @(aligned),
        (id)kIOSurfacePixelFormat: @0});
}

h3_ane_block *h3_ane_block_create(const h3_weight_store *store,
                                  const char *prefix, uint32_t rows,
                                  const uint32_t *segment_ends,
                                  const float *rope_cos_values,
                                  const float *rope_sin_values,
                                  char *error, size_t error_size) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/"
           "AppleNeuralEngine", RTLD_NOW);
    if (!store || !prefix || !rows || !segment_ends || !rope_cos_values ||
        !rope_sin_values) {
        blk_fail(error, error_size, "invalid ANE block arguments");
        return NULL;
    }
    uint32_t padded = (rows + BLK_ROW_MULTIPLE - 1) / BLK_ROW_MULTIPLE *
                      BLK_ROW_MULTIPLE;
    uint32_t ends[H3_ANE_BLOCK_SEGMENTS];
    memcpy(ends, segment_ends, sizeof(ends));
    if (ends[H3_ANE_BLOCK_SEGMENTS - 1] != rows) {
        blk_fail(error, error_size, "segment ends must cover every row");
        return NULL;
    }
    ends[H3_ANE_BLOCK_SEGMENTS - 1] = padded;  /* padding rides the tail */

    size_t bound = 512ull * 1024 * 1024;
    blk_blob blob = {calloc(bound, 1), bound, 64};
    if (!blob.data) {
        blk_fail(error, error_size, "out of memory building the block blob");
        return NULL;
    }
    blob.data[0] = 0x01;
    blob.data[4] = 0x02;

    size_t hr5376 = blk_pack_rotation(&blob, BLK_HIDDEN);
    size_t hr7168 = blk_pack_rotation(&blob, BLK_INNER);
    size_t hr14336 = blk_pack_rotation(&blob, BLK_FFN);
    blk_projection qkv, out, fc1, fc2;
    size_t norm1, norm2, q_norm, k_norm;
    int ok =
        blk_pack_projection(&blob, store, prefix, "attn.qkv_proj",
                            BLK_INNER * 3, BLK_HIDDEN, hr5376, &qkv,
                            error, error_size) &&
        blk_pack_projection(&blob, store, prefix, "attn.out_proj",
                            BLK_HIDDEN, BLK_INNER, hr7168, &out,
                            error, error_size) &&
        blk_pack_projection(&blob, store, prefix, "mlp.fc1",
                            BLK_FFN * 2, BLK_HIDDEN, hr5376, &fc1,
                            error, error_size) &&
        blk_pack_projection(&blob, store, prefix, "mlp.fc2",
                            BLK_HIDDEN, BLK_FFN, hr14336, &fc2,
                            error, error_size) &&
        blk_pack_norm(&blob, store, prefix, "norm1.weight", BLK_HIDDEN,
                      &norm1, error, error_size) &&
        blk_pack_norm(&blob, store, prefix, "norm2.weight", BLK_HIDDEN,
                      &norm2, error, error_size) &&
        blk_pack_norm(&blob, store, prefix, "attn.q_norm.weight",
                      BLK_HEAD_DIM, &q_norm, error, error_size) &&
        blk_pack_norm(&blob, store, prefix, "attn.k_norm.weight",
                      BLK_HEAD_DIM, &k_norm, error, error_size);
    if (!ok) {
        free(blob.data);
        return NULL;
    }
    size_t mask_header = 0;
    if (padded != rows) {
        size_t chunk;
        __fp16 *mask = blk_blob_reserve(&blob, (size_t)padded * 2, &chunk);
        for (uint32_t row = 0; row < padded; row++)
            mask[row] = row < rows ? (__fp16)0.0f : (__fp16)-30000.0f;
        mask_header = chunk;
    }
    size_t rope_header_cos, rope_header_sin;
    __fp16 *cos_packed = blk_blob_reserve(
        &blob, (size_t)padded * BLK_ROPE_DIMS * 2, &rope_header_cos);
    __fp16 *sin_packed = blk_blob_reserve(
        &blob, (size_t)padded * BLK_ROPE_DIMS * 2, &rope_header_sin);
    for (uint32_t row = 0; row < padded; row++)
        for (uint32_t index = 0; index < BLK_ROPE_DIMS; index++) {
            uint32_t half = index % BLK_ROPE_HALF;
            float c = row < rows ?
                rope_cos_values[(size_t)row * BLK_ROPE_HALF + half] : 1.0f;
            float s = row < rows ?
                rope_sin_values[(size_t)row * BLK_ROPE_HALF + half] : 0.0f;
            cos_packed[(size_t)row * BLK_ROPE_DIMS + index] = (__fp16)c;
            sin_packed[(size_t)row * BLK_ROPE_DIMS + index] = (__fp16)s;
        }
    if (blob.cursor > blob.length) {
        blk_fail(error, error_size, "block blob overflow");
        free(blob.data);
        return NULL;
    }

    h3_ane_block *block = calloc(1, sizeof(*block));
    if (!block) {
        blk_fail(error, error_size, "out of memory creating the ANE block");
        free(blob.data);
        return NULL;
    }
    block->rows = rows;
    block->padded_rows = padded;
    block->weight_bytes = blob.cursor;

    @autoreleasepool {
        NSError *failure = nil;
        NSData *weights = [NSData dataWithBytesNoCopy:blob.data
                                               length:blob.cursor
                                         freeWhenDone:YES];
        NSString *source = blk_program(&qkv, &out, &fc1, &fc2, norm1, norm2,
                                       q_norm, k_norm, rope_header_cos,
                                       rope_header_sin, mask_header,
                                       padded != rows, ends,
                                       H3_ANE_BLOCK_SEGMENTS, padded);
        NSData *program = [source dataUsingEncoding:NSUTF8StringEncoding];
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class requestClass = NSClassFromString(@"_ANERequest");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        id descriptor = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
            descriptorClass, @selector(modelWithMILText:weights:optionsPlist:),
            program, @{@"@model_path/weights/weight.bin":
                       @{@"offset": @0, @"data": weights}}, nil);
        if (!descriptor) {
            blk_fail(error, error_size, "ANE block descriptor rejected");
            free(block);
            return NULL;
        }
        id model = ((id(*)(Class, SEL, id))objc_msgSend)(
            modelClass, @selector(inMemoryModelWithDescriptor:), descriptor);
        NSString *identifier = ((id(*)(id, SEL))objc_msgSend)(
            model, @selector(hexStringIdentifier));
        NSString *directory = [NSTemporaryDirectory()
            stringByAppendingPathComponent:identifier];
        NSFileManager *files = [NSFileManager defaultManager];
        [files createDirectoryAtPath:
            [directory stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [program writeToFile:[directory stringByAppendingPathComponent:
            @"model.mil"] atomically:YES];
        [weights writeToFile:[directory
            stringByAppendingPathComponent:@"weights/weight.bin"]
                  atomically:YES];
        block->temporary_directory = strdup(directory.UTF8String);
        double started = blk_seconds();
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(compileWithQoS:options:error:), 21, @{},
                &failure)) {
            blk_fail(error, error_size, "ANE block compile failed: %s",
                     failure ? failure.localizedDescription.UTF8String : "?");
            h3_ane_block_free(block);
            return NULL;
        }
        if (!((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{},
                &failure)) {
            blk_fail(error, error_size, "ANE block load failed: %s",
                     failure ? failure.localizedDescription.UTF8String : "?");
            h3_ane_block_free(block);
            return NULL;
        }
        block->compile_seconds = blk_seconds() - started;
        size_t input_bytes = (size_t)BLK_HIDDEN * padded * sizeof(float);
        size_t output_bytes = (size_t)BLK_HIDDEN * padded * sizeof(float);
        const char *stage_env = getenv("H3_ANE_BLOCK_STAGE");
        if (stage_env) {
            int stage = atoi(stage_env);
            uint32_t widest = BLK_HIDDEN;
            if (stage == 3) widest = BLK_INNER * 3;
            else if (stage == 4 || stage == 5) widest = BLK_INNER;
            else if (stage == 6) widest = BLK_INNER;
            else if (stage == 9) widest = BLK_FFN;
            output_bytes = (size_t)widest * padded * sizeof(float);
        }
        size_t mod_bytes = (size_t)BLK_HIDDEN * H3_ANE_BLOCK_MOD_WIDTH *
                           sizeof(float);
        block->input_surface = blk_surface(input_bytes);
        block->mod_surface = blk_surface(mod_bytes);
        block->output_surface = blk_surface(output_bytes);
        if (!block->input_surface || !block->mod_surface ||
            !block->output_surface) {
            blk_fail(error, error_size, "ANE block surfaces failed");
            h3_ane_block_free(block);
            return NULL;
        }
        block->input_base = IOSurfaceGetBaseAddress(block->input_surface);
        block->mod_base = IOSurfaceGetBaseAddress(block->mod_surface);
        block->output_base = IOSurfaceGetBaseAddress(block->output_surface);
        memset(block->input_base, 0, input_bytes);
        for (uint32_t channel = 0; channel < BLK_HIDDEN; channel++)
            for (uint32_t row = rows; row < padded; row++)
                block->input_base[(size_t)channel * padded + row] = 1.0f;
        memset(block->mod_base, 0, mod_bytes);
        id inputs = @[
            ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:),
                block->input_surface),
            ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
                surfaceClass, @selector(objectWithIOSurface:),
                block->mod_surface)];
        id output = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(
            surfaceClass, @selector(objectWithIOSurface:),
            block->output_surface);
        id request = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(
            requestClass,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                      weightsBuffer:perfStats:procedureIndex:),
            inputs, @[@0, @1], @[output], @[@0], nil, nil, @0);
        if (!request) {
            blk_fail(error, error_size, "ANE block request rejected");
            h3_ane_block_free(block);
            return NULL;
        }
        block->model = (__bridge_retained void *)model;
        block->request = (__bridge_retained void *)request;
    }
    return block;
}

void h3_ane_block_free(h3_ane_block *block) {
    if (!block) return;
    @autoreleasepool {
        if (block->model) {
            id model = (__bridge_transfer id)block->model;
            NSError *failure = nil;
            ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &failure);
        }
        if (block->request) {
            id request = (__bridge_transfer id)block->request;
            (void)request;
        }
        if (block->temporary_directory) {
            [[NSFileManager defaultManager]
                removeItemAtPath:@(block->temporary_directory) error:nil];
            free(block->temporary_directory);
        }
    }
    if (block->input_surface) CFRelease(block->input_surface);
    if (block->mod_surface) CFRelease(block->mod_surface);
    if (block->output_surface) CFRelease(block->output_surface);
    free(block);
}

float *h3_ane_block_input(h3_ane_block *block) {
    return block ? block->input_base : NULL;
}

float *h3_ane_block_mod(h3_ane_block *block) {
    return block ? block->mod_base : NULL;
}

float *h3_ane_block_output(h3_ane_block *block) {
    return block ? block->output_base : NULL;
}

uint32_t h3_ane_block_padded_rows(const h3_ane_block *block) {
    return block ? block->padded_rows : 0;
}

double h3_ane_block_compile_seconds(const h3_ane_block *block) {
    return block ? block->compile_seconds : 0.0;
}

uint64_t h3_ane_block_weight_bytes(const h3_ane_block *block) {
    return block ? block->weight_bytes : 0;
}

int h3_ane_block_eval(h3_ane_block *block, char *error, size_t error_size) {
    if (!block || !block->model || !block->request) {
        blk_fail(error, error_size, "the ANE block is not loaded");
        return 0;
    }
    int ok = 0;
    @autoreleasepool {
        NSError *failure = nil;
        ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
            (__bridge id)block->model,
            @selector(evaluateWithQoS:options:request:error:), 21, @{},
            (__bridge id)block->request, &failure) ? 1 : 0;
        if (!ok)
            blk_fail(error, error_size, "ANE block evaluation failed: %s",
                     failure ? failure.localizedDescription.UTF8String : "?");
    }
    return ok;
}
