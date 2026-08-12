# h3.c-ane — MiniMax H3 video generation on the Apple Neural Engine

This is a fork of [antirez's h3.c](https://github.com/antirez/h3.c) that runs
**every transformer block of the MiniMax H3 video model on the Apple Neural
Engine** of a base 24GB M4 Mac mini. The H3 DiT holds ~19 billion parameters
across its 50 blocks (a 21GB int8 checkpoint) — far more than the machine can
hold in memory, and far more than the ANE was ever meant to chew through.

It works anyway: weights stream from SSD, compiled ANE models rotate through a
small residency window, and the Metal GPU only packs tensors in and out and
decodes the final video.

![hero](assets/h3_ane_hero.gif)

*Prompt: "a hummingbird hovering over a flower" — 4 seconds of video with
audio, 6 denoise steps, generated end-to-end on the Neural Engine of a base
M4 in about 8.5 minutes (DiT 370s, VAE 88s), with the process peaking at
2.1GB of RAM.*

## Why

h3.c already runs H3 beautifully on the GPU with Metal. The point of this fork
is different: to prove that the Neural Engine — a fixed-function accelerator
with no public low-level API, a 16KB alignment rulebook, and an fp16-only
datapath — can run a huge modern diffusion transformer end to end. It is a
proof of concept, not a faster path (yet).

## Quickstart

```sh
# 1. Build (macOS on Apple Silicon)
make h3

# 2. Get the model files (ComfyUI int8 release of MiniMax H3):
#    - minimax_h3_fl2va_pruned_int8_convrot.safetensors  (21GB DiT, int8)
#    - minimax_h3_video_vae_fp16.safetensors
#    - minimax_h3_audio_vae_fp32.safetensors
#    - the MiniMax-H3 tokenizer/config folder (FL2VA)
#    Arrange them as h3.c expects (see README.upstream.md).

# 3. Text conditioning. The text encoder is 50 layers of Qwen3-VL-32B (51GB
#    in BF16) — too big for a 24GB Mac. Mint the conditioning once on any
#    machine with a GPU and disk, then copy the small .h3cd file over:
python3 scripts/mint_h3_conditioning.py \
    --prompt "a hummingbird hovering over a flower" \
    --te qwen3vl_32b_minimax_h3_bf16.safetensors \
    --tokenizer tokenizer.json --out hummingbird.h3cd

# 4. Generate on the Neural Engine:
H3_CONDITIONING_FILE=hummingbird.h3cd \
H3_ANE_FULL_BLOCKS=50 H3_ANE_ROTATE=1 \
./h3 -d MiniMax-H3 -p "a hummingbird hovering over a flower" \
    --width 512 --height 512 --render-width 384 --render-height 384 \
    --seconds 4 --steps 6 --ssd-streaming -o out.mp4
```

The first run compiles all 50 ANE block models (~2s each); they are cached in
`$TMPDIR/h3-ane-cache` (~19GB) so every later run loads them in milliseconds.

## Dependencies

- Apple Silicon Mac (developed and tested on a base M4, 24GB, macOS 26)
- Xcode command line tools (clang, Metal)
- ffmpeg (`brew install ffmpeg`) for the final mp4 mux
- ~45GB free disk: 21GB checkpoint + ~19GB compiled ANE model cache
- For conditioning minting only: any box with Python, torch, safetensors,
  tokenizers, and disk for the 51GB BF16 text encoder

## Environment switches

| Variable | Meaning |
|---|---|
| `H3_ANE_FULL_BLOCKS=N` | Run the first N DiT blocks as whole ANE graphs (50 = all) |
| `H3_ANE_ROTATE=1` | Rotate ANE models through memory (required for large N on 24GB) |
| `H3_ANE_CACHE=0` | Disable the compiled-model cache |
| `H3_CONDITIONING_FILE=x.h3cd` | Use precomputed text conditioning (skips the 51GB text encoder) |
| `H3_PROFILE=1` | Print per-stage timing and memory |
| `H3_ANE_DEBUG_RANGE=1` | Print per-block activation peaks and non-finite counts |

## What actually runs on the Neural Engine

Each of the 50 DiT blocks is one hand-written MIL graph, compiled straight
through the private `AppleNeuralEngine.framework` (no CoreML). One block =
one ANE program containing:

- **int8 weights, dequantized on the fly** — `constexpr` int8 tensors with
  fp16 per-row scales; the ConvRot Hadamard rotation the checkpoint was
  quantized with is applied to activations in-graph (int8 stays int8 in
  memory; nothing is ever materialized as BF16 for the ANE).
- **adaLN modulation** — shift/scale/gate for three modality segments
  (text/audio/video), sliced from one packed input plane.
- **RMSNorm** on a 1/256 pre-scale (the ANE has no `reduce_mean` or `rsqrt`;
  it is `reduce_sum` + `pow(x, -0.5)` with the epsilon folded through).
- **QKV projection + per-head norm + 3-axis RoPE** (48 rotated frequency
  pairs via slice/concat — the ANE has no rotate-half primitive).
- **Full softmax attention, tiled over query rows** — 512-row tiles keep the
  score tensor at `[1,56,512,S]` instead of `[1,56,S,S]`, which is the
  difference between 200MB and 1.3GB of wired memory per block.
- **SwiGLU MLP** with a 1/16 pre/post scale bracket around fc2 (the trained
  model's fc2 output peaks at ±4.3e4 — right at the edge of fp16).
- **Gated residual adds** carrying an adaptive power-of-two range guard:
  the residual stream saturates fp16 from block 0, so the host scales each
  block's input by a power of two, folds 1/s into the gate slots (the
  algebra is exact), and scales the output back. Power-of-two scaling only
  moves the fp16 exponent, so the guard is lossless.

Around the graphs:

- **Model rotation** — a compiled 385M-parameter block is ~800MB wired, so
  50 resident blocks can never fit. Models unload/reload through a
  content-addressed compile cache; the next block's ~250ms disk read hides
  behind the current block's ~280ms ANE evaluation on a prefetch thread.
- **Metal does the rest** — tensor pack/unpack into IOSurfaces, the
  embedders, the final head, and both VAEs. The GPU profile during denoise
  shows `linear=0 conv=0 attention=0`: all transformer math is on the ANE.
- The raw private-framework plumbing lives behind one small interface,
  `h3_ane_bridge.{h,m}` — everything else speaks MIL text and IOSurfaces.

Numerics: a seeded all-ANE render is frame-cosine 0.99 against the pure-Metal
render at 2 steps, and visually identical at 6 steps. The int8/fp16 ANE path
drifts from the BF16 Metal path exactly as much as you'd expect, and no more.

## The Neural Engine is really doing the work

`powermetrics` during denoise:

![powermetrics](assets/powermetrics.png)

## Disclaimer

This is a **proof of concept**, not production software.

- It drives a **private Apple framework** (`AppleNeuralEngine.framework`)
  through `objc_msgSend`. Any macOS update can break it. Nothing here is
  App Store safe.
- **Timing**: on a base M4 at these settings the all-ANE path runs a denoise
  step in 26.3s vs 31.9s for the Metal path (~17% faster) — but that is not
  the point, and bigger GPUs will beat it easily. The point is that it runs
  at all, correctly, on the smallest Apple Silicon machine, while the GPU
  sits at 16 milliwatts.
- **Resources**: expect ~19GB of compiled-model cache on disk, sustained SSD
  reads during generation, and a machine that is genuinely busy. The
  Q-tiled attention and rotation exist because earlier versions wired 12GB+
  of kernel memory and rebooted the machine. A memory watchdog is prudent.
- fp16 is right at the edge for this model. The range guards are measured
  and gated, but other checkpoints may find new cliffs.

## Next steps

- Overlap the Metal pack/unpack with ANE evaluation (they serialize today).
- Native 512² renders (the ANE per-block advantage grows with sequence
  length).
- A second model family through the same bridge, to see which parts
  generalize.

## Thanks

- [antirez](https://github.com/antirez) for h3.c — the cleanest possible
  base to build on, and the reason this fork could focus on the ANE alone.
- MiniMax for the H3 model, and Comfy-Org for the int8 ConvRot
  quantization this port consumes.

## License

MIT, same as upstream h3.c. See `LICENSE`.
