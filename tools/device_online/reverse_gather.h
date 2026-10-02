#pragma once
#include "cann_program.h"
#include <ATen/core/grad_mode.h>
#include <limits>
#include <stdexcept>

namespace tide::device_online {
// One read-only padded source for a single program phase. The recorded copy is
// replayed whenever that phase executes, including device-loop iterations.
// Callers must not mutate the source between this copy and the last select.
// Each owner retains its own indices and destinations; this is not a cache
// across windows/programs or a host-generated numerical batch.
class ReverseGatherInput {
 public:
  ReverseGatherInput(CannProgram& p,const at::Tensor& input):program_(&p),input_(input) {
    if(at::GradMode::is_enabled()||!input.defined()||input.dim()<1||input.size(0)<1
        ||input.size(0)==std::numeric_limits<int64_t>::max()||!input.numel()
        ||input.device().type()!=c10::DeviceType::PrivateUse1||!input.is_contiguous()||input.requires_grad())
      throw std::invalid_argument("reverse gather requires a no-grad contiguous device source");
  }
  void select(CannProgram& p,const at::Tensor& indices,const at::Tensor& output) const {
    if(&p!=program_)throw std::invalid_argument("reverse gather belongs to another program phase");
    // Allocate on the first select, after that owner's original budget checks.
    if(!padded_.defined()) {
      auto shape=input_.sizes().vec();++shape[0];padded_=at::zeros(shape,input_.options());
      p.copy(padded_.narrow(0,0,input_.size(0)),input_);
    }
    p.index_select(padded_,0,indices,output);
  }
 private:
  CannProgram* program_;
  at::Tensor input_;
  mutable at::Tensor padded_;
};
} // namespace tide::device_online
