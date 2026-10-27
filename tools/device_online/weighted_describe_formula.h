#pragma once
#include "weighted_describe.h"
namespace tide::device_online::describe_check {
constexpr int64_t R=6,B=3,D=3,DS=2;
using Fields=std::map<std::string,at::Tensor>;
struct Leaf {std::string name;int64_t row;bool parameter;};
struct Formula {std::vector<at::Tensor> leaves;std::vector<Leaf> labels;std::vector<Fields> fields;};
Formula formula(const Fields&,const at::Tensor& owners,const WeightedDescribeParameters&,const weighted::RegionalConfig&);
}
