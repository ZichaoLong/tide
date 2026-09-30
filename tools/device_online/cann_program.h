#pragma once
#include <ATen/ATen.h>
#include <functional>
#include <memory>
#include <vector>

namespace tide::device_online {
struct CannApi;
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
  void copy(const at::Tensor& destination, const at::Tensor& source);
  void multiply(const at::Tensor&, const at::Tensor&, const at::Tensor& output);
  void softmax(const at::Tensor&, int64_t axis, const at::Tensor& output);
  void sigmoid(const at::Tensor&, const at::Tensor& output);
  void tanh(const at::Tensor&, const at::Tensor& output);
  void relu(const at::Tensor&, const at::Tensor& output);
  void silu(const at::Tensor&, const at::Tensor& output);
  // Normalize the last dimension with unit affine parameters. Per-owner
  // learned affine values are separately packed and applied by the caller.
  void rms_norm(const at::Tensor&, double epsilon, const at::Tensor& output);
  void layer_norm(const at::Tensor&, double epsilon, const at::Tensor& output);
  void batch_matmul(const at::Tensor&,const at::Tensor&,const at::Tensor& output);
  void index_copy(const at::Tensor& target,int64_t axis,const at::Tensor& indices,const at::Tensor& source);
  void equal(const at::Tensor&, const at::Tensor&, const at::Tensor& bool_output);
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
  // Deterministic API-failure injection belongs to the standalone test adapter;
  // no environment switches or runtime mutation of a live program are exposed.
  friend struct CannProgramTestAccess;
  CannProgram(at::Device, const std::function<void(CannApi&)>& configure_api);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace tide::device_online
