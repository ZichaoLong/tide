#pragma once

#include "tide/optimizer.h"
#include <filesystem>
#include <string>

namespace tide {

// A caller-owned file-size admission bound, not a format or model-size limit.
// Raising it also requires enough host memory for the complete codec buffers.
inline constexpr int64_t kDefaultCheckpointMaxBytes = int64_t{1} << 32;

// Native checkpoint files are intentionally independent of Python's torch
// serialization.  The format stores named owner/alias topology, values and an
// optional built-in optimizer continuation under one self-describing schema.
class Checkpoint {
 public:
  static void save(const std::filesystem::path& path, const ParameterRegistry& registry,
                   const NamedOptimizer* optimizer = nullptr,
                   const std::string& identity = "",
                   int64_t max_file_bytes = kDefaultCheckpointMaxBytes);
  static void load(const std::filesystem::path& path, ParameterRegistry& registry,
                   NamedOptimizer* optimizer = nullptr,
                   const std::string& expected_identity = "",
                   int64_t max_file_bytes = kDefaultCheckpointMaxBytes);
};

}  // namespace tide
