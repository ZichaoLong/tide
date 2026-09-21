#pragma once
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace tide {
// Persistent workers, independent of ATen intra-op scheduling.
class NodePool {
 public:
  explicit NodePool(int64_t workers);
  ~NodePool();
  NodePool(const NodePool&) = delete;
  NodePool& operator=(const NodePool&) = delete;
  void run(std::vector<std::function<void()>> jobs);
 private:
  std::vector<std::thread> workers_;
  std::queue<std::packaged_task<void()>> queue_;
  std::mutex mutex_;
  std::condition_variable ready_;
  bool stopping_ = false;
};
}  // namespace tide
