#pragma once
#include "weighted_recurrent.h"
namespace tide::device_online::recurrent_check {
using Fields=std::map<std::string,at::Tensor>;
struct Leaf {std::string name;int64_t row;bool parameter;};
struct Formula {std::vector<at::Tensor> leaves;std::vector<Leaf> labels;std::vector<Fields> fields;at::Tensor metadata;};
Formula formula(const weighted::RecurrentConfig&,WeightedRecurrentAction,const WeightedRawLayout&,
  const std::vector<State>&,const at::Tensor& times,const at::Tensor& mass,const at::Tensor& content,
  const at::Tensor& owners,const at::Tensor& present,const WeightedRecurrentParameters&);
}
