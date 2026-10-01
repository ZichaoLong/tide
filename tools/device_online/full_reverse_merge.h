#pragma once
#include "full_vjp.h"
namespace tide::device_online {
void append_full_reverse_merge(CannProgram&,const at::Tensor& destinations,const at::Tensor& global_nodes,
    const FullVjp& partial,const FullVjp& output,const at::Tensor& source_error,const at::Tensor& error);
}
