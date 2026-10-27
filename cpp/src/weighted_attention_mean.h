#pragma once
#include "tide/types.h"
namespace tide::weighted {
// KV §3–4: nonempty selected positive entries; logits already include log mass.
Tensor attention_mean(const Tensor& logits,const Tensor& values);
}
