#include "portable_torch/runtime.hpp"

#include <ATen/CPUGeneratorImpl.h>

#include <charconv>
#include <cstddef>
#include <limits>
#include <mutex>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <system_error>

#if PORTABLE_TORCH_ENABLE_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <torch/cuda.h>
#endif

namespace portable_torch {
namespace {

constexpr std::string_view kDevicePrefix = "--device=";
constexpr std::string_view kDeviceIndexPrefix = "--device-index=";
constexpr std::string_view kDtypePrefix = "--dtype=";
constexpr std::string_view kSeedPrefix = "--seed=";
constexpr std::string_view kOutputDirPrefix = "--output-dir=";

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

std::string_view value_after(std::string_view argument, std::string_view prefix) {
    const auto value = argument.substr(prefix.size());
    if (value.empty()) {
        throw std::invalid_argument("missing value for " + std::string(prefix));
    }
    return value;
}

std::uint64_t parse_uint64(std::string_view text, std::string_view option_name) {
    std::uint64_t value = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, value, 10);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(
            "invalid nonnegative integer for " + std::string(option_name) + ": " +
            std::string(text));
    }
    return value;
}

std::optional<std::string_view> option_value(
    std::string_view argument,
    std::string_view option_name,
    std::string_view option_prefix,
    int& argument_index,
    int argc,
    char** argv) {
    if (argument == option_name) {
        if (argument_index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + std::string(option_name));
        }
        ++argument_index;
        const std::string_view value(argv[argument_index]);
        if (value.empty()) {
            throw std::invalid_argument("missing value for " + std::string(option_name));
        }
        return value;
    }
    if (starts_with(argument, option_prefix)) {
        return value_after(argument, option_prefix);
    }
    return std::nullopt;
}

torch::ScalarType parse_dtype(std::string_view value) {
    if (value == "auto" || value == "float32") {
        return torch::kFloat32;
    }
    if (value == "float64") {
        return torch::kFloat64;
    }
    if (value == "float16") {
        return torch::kFloat16;
    }
    if (value == "bfloat16") {
        return torch::kBFloat16;
    }
    throw std::invalid_argument(
        "unsupported --dtype='" + std::string(value) +
        "'; expected auto, float32, float64, float16, or bfloat16");
}

int parse_accelerator_index(std::string_view device_spec, std::string_view backend) {
    if (device_spec == backend) {
        return 0;
    }
    const std::string prefix = std::string(backend) + ":";
    if (!starts_with(device_spec, prefix)) {
        throw std::invalid_argument(
            "invalid " + std::string(backend) +
            " device syntax: " + std::string(device_spec));
    }
    const auto parsed = parse_uint64(value_after(device_spec, prefix), "--device");
    if (parsed > static_cast<std::uint64_t>(std::numeric_limits<c10::DeviceIndex>::max())) {
        throw std::invalid_argument(
            std::string(backend) + " logical device index is too large");
    }
    return static_cast<int>(parsed);
}

}  // namespace

RuntimeOptions parse_cli(int argc, char** argv, bool require_explicit_device) {
    RuntimeOptions options;
    std::optional<int> separate_device_index;
    bool saw_device = false;
    bool saw_device_index = false;
    bool saw_dtype = false;
    bool saw_seed = false;
    bool saw_output_dir = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help" || argument == "-h") {
            options.help = true;
            continue;
        }
        if (const auto value = option_value(
                argument, "--device", kDevicePrefix, index, argc, argv)) {
            if (saw_device) {
                throw std::invalid_argument("--device may be specified only once");
            }
            options.device_spec = std::string(*value);
            saw_device = true;
            continue;
        }
        if (const auto value = option_value(
                argument, "--device-index", kDeviceIndexPrefix, index, argc, argv)) {
            if (saw_device_index) {
                throw std::invalid_argument("--device-index may be specified only once");
            }
            const auto parsed = parse_uint64(*value, "--device-index");
            if (parsed > static_cast<std::uint64_t>(
                             std::numeric_limits<c10::DeviceIndex>::max())) {
                throw std::invalid_argument("logical device index is too large");
            }
            separate_device_index = static_cast<int>(parsed);
            saw_device_index = true;
            continue;
        }
        if (const auto value = option_value(
                argument, "--dtype", kDtypePrefix, index, argc, argv)) {
            if (saw_dtype) {
                throw std::invalid_argument("--dtype may be specified only once");
            }
            options.dtype = parse_dtype(*value);
            saw_dtype = true;
            continue;
        }
        if (const auto value = option_value(
                argument, "--seed", kSeedPrefix, index, argc, argv)) {
            if (saw_seed) {
                throw std::invalid_argument("--seed may be specified only once");
            }
            options.seed = parse_uint64(*value, "--seed");
            saw_seed = true;
            continue;
        }
        if (const auto value = option_value(
                argument, "--output-dir", kOutputDirPrefix, index, argc, argv)) {
            if (saw_output_dir) {
                throw std::invalid_argument("--output-dir may be specified only once");
            }
            options.output_dir = std::string(*value);
            saw_output_dir = true;
            continue;
        }
        throw std::invalid_argument("unknown argument: " + std::string(argument));
    }

    if (separate_device_index.has_value()) {
        if (options.device_spec.find(':') != std::string::npos) {
            throw std::invalid_argument(
                "specify a logical index either in --device or --device-index, not both");
        }
        if (options.device_spec == "auto") {
            throw std::invalid_argument(
                "an index with auto is ambiguous; request cuda or npu explicitly");
        }
        if (options.device_spec == "cpu") {
            if (*separate_device_index != 0) {
                throw std::invalid_argument("CPU accepts only logical device index 0");
            }
        } else if (options.device_spec == "cuda" || options.device_spec == "npu") {
            options.device_spec += ":" + std::to_string(*separate_device_index);
        } else {
            throw std::invalid_argument(
                "--device-index requires --device=cpu, --device=cuda, or --device=npu");
        }
    }
    if (require_explicit_device && !saw_device && !options.help) {
        throw std::invalid_argument(
            "--device is required for training/benchmark entry points; "
            "choose a backend explicitly or pass --device auto");
    }
    return options;
}

void print_usage(std::ostream& output, const char* program_name) {
    output
        << "Usage: " << program_name << " [OPTIONS]\n"
        << "  --device auto|cpu|cuda|npu (also accepts --device=... and cuda:N/npu:N)\n"
        << "  --device-index LOGICAL_INDEX\n"
        << "  --dtype auto|float32|float64|float16|bfloat16\n"
        << "  --seed UINT64\n"
        << "  --output-dir PATH (must not already exist)\n"
        << "  --help\n"
        << "\n"
        << "An explicit unavailable accelerator is an error. The generic template does "
           "not compile NPU support.\n";
}

torch::Device resolve_device(const RuntimeOptions& options) {
    const std::string_view requested(options.device_spec);
    if (requested == "auto") {
#if PORTABLE_TORCH_ENABLE_CUDA
        if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
            return torch::Device(torch::kCUDA, 0);
        }
#endif
        return torch::Device(torch::kCPU);
    }

    if (requested == "cpu") {
        return torch::Device(torch::kCPU);
    }

    if (requested == "npu" || starts_with(requested, "npu:")) {
        static_cast<void>(parse_accelerator_index(requested, "npu"));
        throw std::runtime_error(
            "NPU was requested, but this generic template was compiled without an "
            "NPU/torch-npu adapter; build and enable a version-matched libtorch_npu "
            "adapter before retrying");
    }

    if (requested == "cuda" || starts_with(requested, "cuda:")) {
#if PORTABLE_TORCH_ENABLE_CUDA
        const int logical_index = parse_accelerator_index(requested, "cuda");
        if (!torch::cuda::is_available()) {
            throw std::runtime_error(
                "CUDA was explicitly requested, but CUDA is unavailable at runtime");
        }
        const auto device_count = torch::cuda::device_count();
        if (logical_index < 0 || logical_index >= device_count) {
            throw std::runtime_error(
                "CUDA logical device index " + std::to_string(logical_index) +
                " is out of range; visible device count is " + std::to_string(device_count));
        }
        return torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(logical_index));
#else
        throw std::runtime_error(
            "CUDA was requested, but this binary was compiled with "
            "PORTABLE_TORCH_BACKEND=CPU");
#endif
    }

    throw std::invalid_argument(
        "unsupported --device='" + options.device_spec +
        "'; expected auto, cpu, cuda[:logical-index], or npu[:logical-index]");
}

std::string resolution_reason(
    const RuntimeOptions& options, const torch::Device& resolved_device) {
    if (options.device_spec == "auto") {
        if (resolved_device.is_cuda()) {
            return "auto:single-visible-cuda";
        }
        if (resolved_device.is_cpu()) {
            return "auto:no-accelerator-selected-cpu";
        }
        // A future adapter may resolve auto to NPU or another backend. Keep
        // the reason truthful without making the CPU/CUDA starter depend on
        // private backend device types.
        return "auto:resolved-" + resolved_device.str();
    }
    if (resolved_device.is_cpu()) {
        return "explicit:cpu";
    }
    if (resolved_device.is_cuda()) {
        return "explicit:cuda";
    }
    return "explicit:" + resolved_device.str();
}

void seed_runtime(const torch::Device& device, std::uint64_t seed) {
    auto cpu_generator = at::detail::getDefaultCPUGenerator();
    {
        std::lock_guard<std::mutex> lock(cpu_generator.mutex());
        cpu_generator.set_current_seed(seed);
    }
    if (device.is_cpu()) {
        return;
    }
    if (device.is_cuda()) {
#if PORTABLE_TORCH_ENABLE_CUDA
        const c10::cuda::CUDAGuard guard(device);
        torch::cuda::manual_seed(seed);
        return;
#else
        throw std::logic_error("internal error: CUDA device reached a CPU-only binary");
#endif
    }
    throw std::runtime_error(
        "seeding is not implemented for resolved device " + device.str());
}

void synchronize(const torch::Device& device) {
    if (device.is_cpu()) {
        return;
    }
    if (device.is_cuda()) {
#if PORTABLE_TORCH_ENABLE_CUDA
        torch::cuda::synchronize(device.index());
        return;
#else
        throw std::logic_error("internal error: CUDA device reached a CPU-only binary");
#endif
    }
    throw std::runtime_error(
        "synchronization is not implemented for resolved device " + device.str());
}

const char* compiled_backend() noexcept {
#if PORTABLE_TORCH_ENABLE_CUDA
    return "cuda";
#else
    return "cpu";
#endif
}

std::string dtype_name(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32:
            return "float32";
        case torch::kFloat64:
            return "float64";
        case torch::kFloat16:
            return "float16";
        case torch::kBFloat16:
            return "bfloat16";
        default:
            return c10::toString(dtype);
    }
}

}  // namespace portable_torch
