#pragma once
// KV §2–4: multirow packing target (single rows may exceed it), int64 times.
#include "weighted_query_batch.h"
namespace tide::weighted::query {
std::vector<std::vector<Index>> snapshot_chunks(const States&,Index width,Index budget=32<<20);
struct SnapshotLayout {
  States states;
  Tensor indices,visible,age;
  Index rows,length;
  SnapshotLayout(States,const Times&,Index window);
  Tensor slot(const std::string&) const;
  Tensor visible_slot(const std::string&) const;
  Tensor project_slot(const std::string&,const Tensor&) const;
  Tensor weights() const;
};
std::vector<std::string> packed_branches(const States&,const Times&,const KVConfig&,bool is_full);
std::array<Tensor,4> packed_memory(const NodeWeights&,const States&,const Times&,const Tensor&,const KVConfig&);
Tensor packed_full(const NodeWeights&,const States&,const Times&,const Tensor&,const KVConfig&);
}
