#pragma once
// Benchmark complete-return transport; no model, checkpoint or live-state edits.
#include <ATen/ATen.h>
#include <vector>
namespace weighted_benchmark::return_copy {
constexpr int64_t kPackBytes=8*1024*1024;
struct Row {at::Tensor value;std::vector<at::Tensor*> outputs;};
// A bounded compatible group. Exposed here for literal CPU/device copy checks.
void group(const std::vector<Row>&);
void tensors(const std::vector<at::Tensor*>&,bool packed);
}
