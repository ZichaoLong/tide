#pragma once
#include <ATen/ATen.h>
#include <functional>
#include <memory>
#include <vector>

namespace tide::device_online {
// Experimental single-device control component, not a graph executor. It owns
// model/stream/label/descriptor/workspace lifetimes. Operations below affect
// fixed-size control buffers only; no autograd or hidden CPU fallback.
class CannProgram {
 public:
  explicit CannProgram(at::Device);
  ~CannProgram();
  CannProgram(const CannProgram&) = delete;
  CannProgram& operator=(const CannProgram&) = delete;
  size_t label();
  void mark(size_t);
  void branch(const at::Tensor& int32_index, const std::vector<size_t>& labels);
  void add(const at::Tensor& destination, const at::Tensor& increment);
  void multiply(const at::Tensor&, const at::Tensor&, const at::Tensor& output);
  void index_select(const at::Tensor&, int64_t axis, const at::Tensor& int64_indices, const at::Tensor& output);
  // Submit a backend kernel once at model construction; its device task is
  // replayed by runtime control flow. Retain all buffers through completion.
  void kernel(std::function<void(void* stream)>, const std::vector<at::Tensor>& buffers);
  void less(const at::Tensor& int64_left, const at::Tensor& int64_right, const at::Tensor& bool_output);
  void logical_and(const at::Tensor&, const at::Tensor&, const at::Tensor& bool_output);
  void cast_index(const at::Tensor& bool_input, const at::Tensor& int32_output);
  void finish();
  // One submission and one explicit boundary wait. All intervening choices are
  // device tasks. Caller must finish input writes before run and keep capacity
  // checks in its device program. Runtime timeout is an execution error.
  void run(int32_t timeout_ms = 10000);
  void close();
  int64_t workspace_bytes() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace tide::device_online
