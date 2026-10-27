#pragma once
#include "tide/weighted.h"
namespace tide::weighted::batch {
Tensor attention(const Tensor& scores,const Tensor& caps,const Tensor& received,
                 const std::string& mode,const std::string& base,const Tensor& temperature);
Tensor proportional(const Tensor& caps,Index k);
Tensor sorted_send(const Tensor& scores,const Tensor& caps,const Tensor& nodes,
                   const Tensor& target,Index k=0);
}
