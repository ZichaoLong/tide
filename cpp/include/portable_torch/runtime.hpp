#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>

#include <torch/torch.h>

#ifndef PORTABLE_TORCH_ENABLE_CUDA
#define PORTABLE_TORCH_ENABLE_CUDA 0
#endif

namespace portable_torch {

struct RuntimeOptions {
    std::string device_spec = "auto";
    torch::ScalarType dtype = torch::kFloat32;
    std::uint64_t seed = 0;
    std::string output_dir;
    bool help = false;
};

// Pass require_explicit_device=true for training/benchmark entry points.
// The bundled demo keeps the low-risk diagnostic default of auto.
RuntimeOptions parse_cli(
    int argc, char** argv, bool require_explicit_device = false);
void print_usage(std::ostream& output, const char* program_name);

torch::Device resolve_device(const RuntimeOptions& options);
std::string resolution_reason(
    const RuntimeOptions& options, const torch::Device& resolved_device);
void seed_runtime(const torch::Device& device, std::uint64_t seed);
void synchronize(const torch::Device& device);

const char* compiled_backend() noexcept;
std::string dtype_name(torch::ScalarType dtype);

}  // namespace portable_torch
