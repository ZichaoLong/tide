#pragma once
#include "tide/checkpoint.h"
namespace tide {
// Complete native numerical checkpoint: all named model values/aliases,
// optional optimizer, exact raw State/History/pending/input ledger and int64
// clocks. This is an explicit autograd boundary, independent of Python.
class GraphCheckpoint {
 public:
  static void save(const std::filesystem::path&,const Graph&,const Model&,
                   const Continuation&,const NamedOptimizer* optimizer=nullptr,
                   int64_t max_file_bytes=kDefaultCheckpointMaxBytes);
  // Validate the whole record and allocate owner transfers before mutation.
  // Destinations retain parameter identity; returned state is detached.
  static Continuation load(const std::filesystem::path&,const Graph&,Model&,
                           NamedOptimizer* optimizer=nullptr,
                           int64_t max_file_bytes=kDefaultCheckpointMaxBytes);
};
}
