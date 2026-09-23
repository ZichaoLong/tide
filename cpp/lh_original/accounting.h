#pragma once
// Injected only into the explicitly instrumented, project-owned LH copy.
#include "tide/operator_work.h"
#include "tide/operator_profile.h"
#include "portable_torch/threads.hpp"
#include <chrono>
#include <cstdlib>

namespace lh_accounting {
inline bool profiling() {
  const char* value = std::getenv("TIDE_LH_OPERATOR_PROFILE");
  return value && std::string(value) == "1";
}
inline bool counting() {
  const char* value = std::getenv("TIDE_LH_WORK");
  return value && std::string(value) == "1";
}
inline void initialize() {
  const char* seed = std::getenv("TIDE_LH_SEED");
  torch::manual_seed(seed ? std::stoll(seed) : 7);
  std::cout << "RUNTIME " << nlohmann::json(portable_torch::thread_metrics()).dump()
            << '\n' << portable_torch::blas_description() << std::endl;
}
inline int64_t pending(const BaseCortexNet& net, const VPtrBatchSignals& signals) {
  int64_t rows = 0;
  for (size_t i = 0; i < signals.size(); ++i) if (signals[i])
    rows += signals[i]->x.size(0)*(net.A.indptr[i+1]-net.A.indptr[i]);
  return rows;
}
inline void finish(int64_t token, const Tensor& logits, const IOCortexNet& net,
                   const VPtrBatchSignals& iacts, const VPtrBatchSignals& oacts,
                   double elapsed) {
  auto metrics = counting() ? tide::work::metrics() : std::map<std::string, double>{};
  if (profiling()) {
    auto detail = tide::op_profile::metrics(); metrics.insert(detail.begin(), detail.end());
  }
  if (counting()) metrics["op/pending_edge_rows"] = pending(*net.inet, iacts)+
    pending(*net.onet, oacts)+pending(*net.iobridge, iacts)+pending(*net.oibridge, oacts);
  metrics["check/logits_sum"] = logits.detach().to(torch::kFloat64).sum().item<double>();
  metrics["perf/token_seconds"] = elapsed;
  auto runtime = portable_torch::thread_metrics(); metrics.insert(runtime.begin(), runtime.end());
  nlohmann::json event{{"step", token}, {"metrics", metrics}};
  if (std::getenv("TIDE_LH_AUDIT")) {
    if (logits.numel() > 4096) throw std::runtime_error("audit logits restricted to small fixtures");
    auto flat = logits.detach().to(torch::kFloat64).contiguous().view({-1});
    auto p = flat.data_ptr<double>(); event["logits"] = std::vector<double>(p, p+flat.numel());
  }
  std::cout << "WORK " << event.dump() << std::endl;
}
} // namespace lh_accounting
