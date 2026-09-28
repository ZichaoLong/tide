#include "tide/pool.h"
#include <ATen/ThreadLocalState.h>
#include <c10/core/StreamGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <stdexcept>

namespace tide {
NodePool::NodePool(int64_t workers) {
  if (workers < 1 || workers > 256) throw std::invalid_argument("workers must be in [1,256]");
  if (workers == 1) return;
  for (int64_t i = 0; i < workers; ++i) {
    workers_.emplace_back([this] {
      for (;;) {
        std::packaged_task<void()> task;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
          if (stopping_ && queue_.empty()) return;
          task = std::move(queue_.front()); queue_.pop();
        }
        task();
      }
    });
  }
}
NodePool::~NodePool() {
  { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
  ready_.notify_all();
  for (auto& worker : workers_) worker.join();
}
void NodePool::run(std::vector<std::function<void()>> jobs) {
  if (workers_.empty() || jobs.size() < 2) {
    for (auto& job : jobs) job();
    return;
  }
  const at::ThreadLocalState caller;
  std::optional<c10::Stream> stream;
  if (device_ && !device_->is_cpu()) {
    c10::impl::VirtualGuardImpl implementation(device_->type());
    stream = implementation.getStream(*device_);
  }
  std::vector<std::future<void>> futures;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& job : jobs) {
      std::packaged_task<void()> task([caller, stream, job = std::move(job)] {
        at::ThreadLocalStateGuard guard(caller);
        c10::OptionalStreamGuard stream_guard(stream);
        job();
      });
      futures.push_back(task.get_future());
      queue_.push(std::move(task));
    }
  }
  ready_.notify_all();
  std::exception_ptr error;
  // Drain all tasks even on failure: their captures may reference caller storage.
  for (auto& future : futures) {
    try { future.get(); } catch (...) { if (!error) error = std::current_exception(); }
  }
  if (error) std::rethrow_exception(error);
}
}  // namespace tide
