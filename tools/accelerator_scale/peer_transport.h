#pragma once
#include <ATen/ATen.h>
#include <memory>
#include <vector>
#include <optional>

namespace accelerator_scale {
// One process-local replay session, with explicit finite notification capacity.
// Warmup creates two reusable IPC notifications per directed device pair.
// Each copy includes a consumed acknowledgement, ordering source-buffer reuse.
// Capture verifies the finite warmup transfer inventory, independent of size.
class PeerTransport {
 public:
  explicit PeerTransport(const std::vector<at::Device>&, size_t capacity=16384);
  ~PeerTransport();
  void prepare_capture();
  void finish_capture();
  std::pair<size_t,int64_t> inventory() const;
  at::Tensor copy(const at::Tensor&, at::Device);
  std::vector<at::Tensor> vjp(const at::Tensor&,const std::vector<at::Tensor>&,const at::Tensor&,bool retain);
  PeerTransport(const PeerTransport&)=delete;
  PeerTransport& operator=(const PeerTransport&)=delete;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
at::Tensor replay_peer_copy(const at::Tensor&, at::Device);
std::optional<std::vector<at::Tensor>> replay_peer_vjp(const at::Tensor&,
    const std::vector<at::Tensor>&,const at::Tensor&,bool retain);
}  // namespace accelerator_scale
