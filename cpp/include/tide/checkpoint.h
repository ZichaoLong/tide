#pragma once

#include "tide/optimizer.h"
#include <filesystem>
#include <string>

namespace tide {

// Native checkpoint files are intentionally independent of Python's torch
// serialization.  The format stores named owner/alias topology, values and an
// optional built-in optimizer continuation under one self-describing schema.
class Checkpoint {
 public:
  static void save(const std::filesystem::path& path, const ParameterRegistry& registry,
                   const NamedOptimizer* optimizer = nullptr,
                   const std::string& identity = "");
  static void load(const std::filesystem::path& path, ParameterRegistry& registry,
                   NamedOptimizer* optimizer = nullptr,
                   const std::string& expected_identity = "");
};

}  // namespace tide
