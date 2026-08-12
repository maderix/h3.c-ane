/* Decode a random-normal latent through the video VAE: a healthy decoder
 * renders colorful structure, a broken weight mapping renders a flat field.
 * usage: h3_vae_noise_test VAE_DIR [OUT.ppm] */

#include "h3_video_vae.h"
#include "h3_host.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum { LATENT_T = 7, LATENT_H = 16, LATENT_W = 16, CHANNELS = 24 };

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s VAE_DIR [OUT.ppm]\n", argv[0]); return 2; }
    size_t elements = (size_t)CHANNELS * LATENT_T * LATENT_H * LATENT_W;
    float *latent = malloc(elements * sizeof(*latent));
    h3_rng rng;
    h3_rng_seed(&rng, 11);
    h3_rng_fill_normal(&rng, latent, elements);
    h3_video_frames frames = {0};
    char error[512] = {0};
    if (!h3_video_vae_decode(argv[1], "h3_shaders.metal", latent, LATENT_T,
                             LATENT_H, LATENT_W, NULL, NULL, &frames,
                             error, sizeof(error))) {
        fprintf(stderr, "decode failed: %s\n", error);
        return 1;
    }
    size_t pixels = (size_t)frames.height * frames.width * 3;
    const float *mid = frames.rgb + (size_t)(frames.frames / 2) * pixels;
    double mean = 0, sq = 0;
    for (size_t index = 0; index < pixels; index++) {
        mean += mid[index];
        sq += (double)mid[index] * mid[index];
    }
    mean /= pixels;
    double std = sqrt(sq / pixels - mean * mean);
    printf("frames=%d %dx%d mid-frame mean=%.4f std=%.4f %s\n",
           frames.frames, frames.width, frames.height, mean, std,
           std > 0.05 ? "STRUCTURED" : "FLAT");
    if (argc > 2) {
        FILE *out = fopen(argv[2], "wb");
        if (out) {
            fprintf(out, "P6\n%d %d\n255\n", frames.width, frames.height);
            for (size_t index = 0; index < pixels; index++) {
                float value = mid[index];
                if (value < 0) value = 0;
                if (value > 1) value = 1;
                fputc((int)(value * 255.0f), out);
            }
            fclose(out);
        }
    }
    h3_video_frames_free(&frames);
    free(latent);
    return 0;
}
