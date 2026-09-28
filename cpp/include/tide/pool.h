#pragma once
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <optional>
#include <c10/core/Device.h>

namespace tide {
// Persistent workers, independent of ATen intra-op scheduling.
class NodePool {
 public:
  explicit NodePool(int64_t workers);
  ~NodePool();
  NodePool(const NodePool&) = delete;
  NodePool& operator=(const NodePool&) = delete;
  void run(std::vector<std::function<void()>> jobs);
  void set_device(c10::Device device) { device_ = device; }
 private:
  std::optional<c10::Device> device_;
  std::vector<std::thread> workers_;
  std::queue<std::packaged_task<void()>> queue_;
  std::mutex mutex_;
  std::condition_variable ready_;
  bool stopping_ = false;
};
}  // namespace tide
