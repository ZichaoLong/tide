#pragma once
// Typed benchmark/qualification diagnostics, independent of Python and JSON.
#include "tide/types.h"
#include <stdexcept>

namespace tide_bench {
enum class DifferenceKind { numeric, discrete, structure, nonfinite };
const char* difference_name(DifferenceKind);
struct Difference { DifferenceKind kind; std::string path, message; };
class ComparisonFailure : public std::runtime_error {
 public:
  Difference difference;
  explicit ComparisonFailure(Difference value);
};
struct ComparisonReport {
  size_t numeric = 0, discrete = 0, limit = 32;
  std::vector<Difference> differences;
  bool passed() const { return numeric == 0 && discrete == 0; }
  void add(Difference);
  void merge(const ComparisonReport&);
};
void check_finite(const tide::Result&, bool traces = true);
void check_finite(const std::map<std::string, at::Tensor>&, const std::string& path = "tensors");
ComparisonReport collect_compare(const tide::Result&, const tide::Result&, bool traces,
    at::ScalarType payload_dtype = at::kDouble,
    std::optional<at::Device> candidate_device = std::nullopt,
    double rtol = -1., double atol = -1., size_t max_differences = 32);
ComparisonReport collect_tensors(const std::map<std::string, at::Tensor>&,
    const std::map<std::string, at::Tensor>&, at::ScalarType payload_dtype,
    const std::string& path, double rtol = -1., double atol = -1., size_t max_differences = 32);
}  // namespace tide_bench
