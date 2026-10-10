#!/usr/bin/env python3
"""Generate CUDA/CPU semantic-source adapters and backend-neutral launch headers.

No input/trajectory specialization. Every registered kernel must translate;
unsupported syntax or missing registrations fail the build. CPU adapters only
exercise semantic source/primitive contracts and never implement CUDA fallback.
"""
import argparse
from pathlib import Path
import re


def generate(source, output, backend):
    output.mkdir(parents=True, exist_ok=True)
    registry = re.findall(r"ascendc_library\((\w+) STATIC (\w+\.cpp)\)", (source / "CMakeLists.txt").read_text())
    if {name for _, name in registry} != {p.name for p in source.glob("tide_*.cpp")}:
        raise ValueError("device kernel inventory differs from registered semantic sources")
    targets = []
    for target, name in registry:
        text = (source / name).read_text()
        declarations = re.findall(r'extern "C" __global__ __aicore__ void (\w+)\s*\(([^)]*)\)\s*\{', text)
        if len(declarations) != 1:
            raise ValueError("unsupported kernel declaration: " + name)
        symbol, arguments = declarations[0]
        args = [a.strip() for a in arguments.replace("\n", " ").split(",")]
        if any(not re.fullmatch(r"(GM_ADDR|int64_t|uint64_t|int32_t|uint32_t|float|double)\s+\w+", a) for a in args):
            raise ValueError("unsupported kernel argument: " + name)
        ids = [a.split()[-1] for a in args]
        public = arguments.replace("GM_ADDR", "void*")
        header = '#pragma once\n#include <cstdint>\n'
        if backend == "npu":
            header += f'#include "aclrtlaunch_{symbol}.h"\n#define TIDE_LAUNCH_KERNEL ACLRT_LAUNCH_KERNEL\n'
        else:
            header += f'extern "C" int tide_cuda_launch_{symbol}(uint32_t,void*,{public});\n'
            header += '#define TIDE_LAUNCH_KERNEL(name) tide_cuda_launch_##name\n'
            text = text.replace('__global__ __aicore__', '__global__')
            text = re.sub(r'(extern "C" __global__ void \w+\s*\([^)]*\)\s*\{)', r'\1\n  tide_cuda::reset_arena();', text)
            casts = [f'static_cast<GM_ADDR>({i})' if a.startswith("GM_ADDR") else i for a, i in zip(args, ids)]
            wrapper = f'\nextern "C" int tide_cuda_launch_{symbol}(uint32_t blocks,void* stream,{public}) {{\n'
            wrapper += '  if(!blocks)return 1;\n'
            if backend == "cpu":
                text = '#define TIDE_KERNEL_CPU\n' + text
                wrapper += f'  tide_cuda::workers=blocks;for(tide_cuda::worker=0;tide_cuda::worker<blocks;++tide_cuda::worker){symbol}({",".join(casts)});return 0;\n}}\n'
            else:
                wrapper += f'  {symbol}<<<blocks,1,tide_cuda::scratch_bytes,static_cast<cudaStream_t>(stream)>>>({",".join(casts)});return int(cudaGetLastError());\n}}\n'
            (output / (symbol + (".cpp" if backend == "cpu" else ".cu"))).write_text(text + wrapper)
        (output / f"device_launch_{symbol}.h").write_text(header)
        targets.append((target, symbol))
    if backend != "npu":
        extension = ".cpp" if backend == "cpu" else ".cu"
        cmake = [f'add_library({t} STATIC "{output / (s + extension)}")\nset_target_properties({t} PROPERTIES POSITION_INDEPENDENT_CODE ON)\n' for t, s in targets]
        (output / "targets.cmake").write_text("".join(cmake))
    return targets


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("cpu", "cuda", "npu"), required=True)
    args = p.parse_args()
    generate(args.source.resolve(), args.output.resolve(), args.backend)
