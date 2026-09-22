#pragma once
#include "tide/checkpoint.h"
#include <cstdint>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Private persistence layers. Public callers use tide/checkpoint.h only.
namespace tide::checkpoint_detail {
constexpr uint64_t kMaxFileBytes = 1ULL << 32;
constexpr uint64_t kMaxCollection = 1000000;
constexpr uint64_t kMaxTensorElements = 1ULL << 28;
[[noreturn]] inline void fail(const std::string& message) {
  throw std::invalid_argument("native checkpoint: " + message);
}
struct DecodedOwner { std::vector<std::string> aliases; Tensor value; };
struct Decoded {
  std::string identity;
  std::vector<DecodedOwner> owners;
  bool has_optimizer = false;
  std::string optimizer_class;
  std::vector<OptimizerGroup> groups;
  std::map<std::string, OptimizerState> state;
};
void validate_value(const Tensor& value, const std::string& where);
std::map<std::string, Tensor> owner_values(const ParameterRegistry& registry);
void validate_decoded(const Decoded&, const ParameterRegistry&, const NamedOptimizer*,
                      const std::string& expected_identity);
std::vector<uint8_t> encode(const Decoded&);
Decoded decode(const std::vector<uint8_t>&);
std::vector<uint8_t> read_all(const std::filesystem::path&);
void publish_exclusive(const std::filesystem::path&, const std::vector<uint8_t>&);
}  // namespace tide::checkpoint_detail
