#pragma once
#include "benchmark.h"
#include "tide/weighted_column_checkpoint.h"
namespace weighted_benchmark {
// Static fixture-to-columns preparation is not a numerical execution/replay.
// The hot path clones/transfers actual tensors by column and never builds events.
class EagerColumns {
  weighted::columns::WindowExecutor executor_;
  weighted::columns::Layout layout_;
  std::map<Index,Tensor> values_;
  std::map<Index,std::vector<Index>> rows_;
  std::map<Index,at::Device> devices_;
  Index positions_;
 public:
  EagerColumns(SettleGraph,const Fixture&,ExecutionPlacement);
  weighted::columns::WindowResult run(const Continuation&,Index index,bool training) const;
  const SettleGraph& spec()const{return executor_.spec();}
};
std::unique_ptr<EagerColumns> column_owner(const Json&,const Fixture&,bool prefill,bool scalar,
    const Options&,ExecutionPlacement,const std::string& returns,std::string& fallback);
Tensor column_loss(const weighted::columns::Result&,bool allow_empty,double scale);
Json column_work(const weighted::columns::WindowResult&);
Result column_projected(const SettleGraph&,const weighted::columns::WindowResult&,const std::string&);
}
