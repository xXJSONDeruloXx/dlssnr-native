# dlssnr-native

Native Vulkan backend development for DLSS Neural Rendering on RDNA4.
**Development preview, not a playable DLSSNR implementation yet.**

Includes `libdlssnr-native.so`, a PTX compiler frontend, and a model reader.
The standalone library passes 512/512 FP16/FP8 matrix-layout checks on an
RX 9070 XT. Full-model scheduling, synchronization, and gameplay are unvalidated;
104 kernels still stop at unsupported memory-barrier operations.
No HIP/ROCm runtime or third-party executable is required.

User-provided model: **NVIDIA DLSSNR 310.8.0.0**, tested filename
`nvngx_dlssnr.approx-fp16-sm_75-sm_86-sm_89-sm_120.dll`, SHA-256:

```text
dcc0dc2414aedec4a8e084647070383be068554042587180c20c784d4772d36f
```

Validate the model without executing it:
`python3 -m native.model /path/to/nvngx_dlssnr.dll`

Build with a C compiler, Vulkan headers/loader, Python 3.9+, and `glslc`:
`make`. Run source tests with `make test` and the GPU check with `make check`.
Extracted Linux x86-64 releases include the library and test SPIR-V, so
`make check` needs only Python, the Vulkan loader, and a supported AMD GPU.
Tested on SteamOS 3.8; other systems are unverified.

GPLv3. Model bytes and recovered GPU code are not distributed.
