#pragma once
#include "device_program.h"
#include "resident_storage_budget.h"
namespace tide::device_online {
inline int64_t weighted_storage_bytes(const std::vector<at::Tensor>& tensors){
  ResidentStorageBytes ledger;
  for(const auto& x:tensors)if(x.defined())ledger.add(x.storage().unsafeGetStorageImpl(),x.storage().nbytes());
  return ledger.bytes();
}
// Local allowance measures newly retained storage. Arguments belong to the
// caller's owner budget even on their first use in this particular program.
// Freeze the borrowed set before construction: adding local temporaries later
// would incorrectly exempt allocations that this scope must charge.
class WeightedStorageIncrement {
 public:
  WeightedStorageIncrement(DeviceProgram& program,std::vector<at::Tensor> borrowed)
      :program_(program),borrowed_(std::move(borrowed)),
       initial_(program.retained_tensor_bytes(borrowed_)),local_initial_(program.retained_tensor_bytes()){}
  int64_t bytes()const{return program_.retained_tensor_bytes(borrowed_)-initial_;}
  int64_t local_delta()const{return program_.retained_tensor_bytes()-local_initial_;}
 private:
  DeviceProgram& program_;
  std::vector<at::Tensor> borrowed_;
  int64_t initial_,local_initial_;
};
}
