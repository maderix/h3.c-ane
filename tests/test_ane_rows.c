/* Which row counts does an ANE 1x1-convolution graph accept?
 * usage: h3_ane_rows_test FIRST LAST STEP [K] [N] */

#include "h3_ane_linear.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    uint32_t first = argc > 1 ? (uint32_t)atoi(argv[1]) : 1;
    uint32_t last = argc > 2 ? (uint32_t)atoi(argv[2]) : 64;
    uint32_t step = argc > 3 ? (uint32_t)atoi(argv[3]) : 1;
    uint32_t input_dim = argc > 4 ? (uint32_t)atoi(argv[4]) : 1024;
    uint32_t output_dim = argc > 5 ? (uint32_t)atoi(argv[5]) : 256;
    if (!h3_ane_linear_available()) return 2;
    uint16_t *weights = malloc((size_t)input_dim * output_dim * 2);
    if (!weights) return 2;
    for (size_t i = 0; i < (size_t)input_dim * output_dim; i++)
        weights[i] = 0x3c00u;  /* fp16 1.0 */
    for (uint32_t rows = first; rows <= last; rows += step) {
        char error[256] = {0};
        h3_ane_linear *linear = h3_ane_linear_create(
            "probe", weights, H3_ANE_W_F16, input_dim, output_dim, rows,
            h3_ane_linear_default_chunk(input_dim), error, sizeof(error));
        if (!linear) { printf("%5u create %s\n", rows, error); continue; }
        uint32_t plane_rows = h3_ane_linear_plane_rows(linear);
        float *in = h3_ane_linear_input(linear, 0);
        for (uint32_t k = 0; k < input_dim; k++)
            for (uint32_t r = 0; r < plane_rows; r++)
                in[(size_t)k * plane_rows + r] = r < rows ? 1.0f : 0.0f;
        int ok = h3_ane_linear_eval(linear, error, sizeof(error));
        float first_out = ok ? h3_ane_linear_output(linear)[0] : 0.0f;
        float last_out = ok ? h3_ane_linear_output(linear)[
            (size_t)(output_dim - 1) * plane_rows + rows - 1] : 0.0f;
        printf("%5u %s first=%.1f last=%.1f expected=%u%s\n", rows,
               ok ? "ok  " : "FAIL", first_out, last_out, input_dim,
               ok ? "" : " <- eval rejected");
        fflush(stdout);
        h3_ane_linear_free(linear);
    }
    free(weights);
    return 0;
}
