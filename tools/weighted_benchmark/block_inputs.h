#pragma once
#include "benchmark.h"
#include "tide/settle.h"
namespace weighted_benchmark {
struct BlockInputs {Tensor positions,payload,present;};
BlockInputs block_inputs(const SettleGraph&,const Fixture&);
struct OutputObjective {Tensor value,strength_root,content_root;};
OutputObjective block_objective(const Tensor& strength,const Tensor& content,const Tensor& present,double scale,bool roots);
std::vector<Output> block_outputs(const SettleGraph&,Index length,const Tensor& strength,const Tensor& content,
                                 const Tensor& present,const Tensor& times);
}
