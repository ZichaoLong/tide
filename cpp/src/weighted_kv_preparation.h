#pragma once
#include "tide/types.h"
namespace tide::weighted {
std::vector<std::map<std::string,Tensor>> prepare_private_kv(
    const std::vector<const NodeWeights*>&,const std::map<std::string,Tensor>&,
    std::map<std::string,Index>&);
}
