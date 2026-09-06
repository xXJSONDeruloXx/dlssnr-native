# dlssnr-native

Native Linux shared-runtime development for DLSS Neural Rendering.
**Development preview: model loading and Vulkan runtime plumbing are working;
the reconstructed Stage-1 graph is not yet executed by this library.**

The current direction uses the independently reconstructed high-level model
semantics from `dlssnr-portable-research` instead of requiring NVIDIA PTX to
be the production execution format. The older PTX compiler/probes remain in
this repository as research and validation tooling.

## Current Linux shared object

`native/build/libdlssnr-native.so` exposes ABI v1 for:

- opening a supported user-provided `nvngx_dlssnr.dll` directly without
  loading or executing the PE;
- opening an optional local `DLSSNRM1` model package;
- fail-closed DLL/model fingerprint checks and per-tensor SHA-256 validation;
- direct access to the 153 verified model tensors;
- recovered FP8 E4M3FN, FP16, Tensor Core matrix-layout and TIN decoding;
- Vulkan runtime creation and device discovery;
- binding a verified model to a runtime; and
- discovering the qualified Stage-1 profile: RGB, 1280x720, frame 0,
  reset=true.

`dlssnr_stage1_evaluate_rgb_f32()` is part of the ABI but currently returns
`DLSSNR_ERROR_UNSUPPORTED`. This branch does not claim native image inference
until the recovered graph is scheduled and numerically validated through that
entry point.

## Supported model for the new runtime

The new high-level runtime path is fingerprint-locked to the same source model
as `dlssnr-portable-research`:

```text
nvngx_dlssnr.dll
SHA-256: 6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927f
Tensors: 153
Weight fingerprint:
c7f86ab233356fe73d7f66b749559c85838cdce3149b228480154801dff8bc60
```

The DLL is user-provided and is parsed as data only. It is not distributed by
this project. The runtime can use it directly; generating a separate model
cache is optional.

The older PTX development path and `native/model.py` are pinned to a different
tested 310.8 DLL hash (`dcc0dc...`). Those two model builds are intentionally
not treated as interchangeable until their normalized tensor inventories and
graph compatibility are verified.

## Build and inspect

Requirements for the shared runtime are a C compiler and Vulkan headers/loader.
The existing matrix-layout probe additionally uses `glslc`.

```sh
make -C native

native/build/dlssnr-inspect /path/to/nvngx_dlssnr.dll
native/build/dlssnr-inspect --device /path/to/nvngx_dlssnr.dll
```

The first command checks the model without creating a Vulkan device. The
`--device` form also creates the Linux Vulkan runtime and prints the selected
AMD device.

Run host-only tests with:

```sh
make test
```

These compile the C model parsers and recovered numeric/layout decoder with
warnings-as-errors and do not require a GPU. Optional private fixtures can be
supplied through `DLSSNR_PORTABLE_TEST_MODEL` and `DLSSNR_TEST_PACKAGE`.

The existing GPU matrix-layout check remains:

```sh
make
make check
```

It previously passed 512/512 FP16/FP8 matrix-layout checks on an RX 9070 XT.
That result validates the low-level Vulkan/layout probe, not the new full
high-level model scheduler.

## Runtime architecture

```text
user nvngx_dlssnr.dll
          |
          | bounded parse + fingerprint validation
          v
   libdlssnr-native.so
          |
          | recovered model graph (being implemented)
          v
      Vulkan compute
          |
          v
        RADV
```

A Proton integration can keep the raw model on the Linux side and connect the
Windows NGX proxy to a native host. The current control plane is implemented on
the `feat/linux-shared-runtime` branches of `dlssnr-proton` and
`dlss5-linux-bridge`. Frame/resource transport is deliberately not advertised
until native graph evaluation exists.

No HIP/ROCm runtime is required by this shared-object path. Model bytes,
generated model packages, NVIDIA DLLs, and recovered proprietary GPU code are
not distributed.

GPLv3.
