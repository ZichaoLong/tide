#pragma once
// Installed weighted functions -> numerical role banks. Architecture §2–4.
#include "weighted_describe.h"
#include "weighted_recurrent.h"
#include "weighted_kv_update.h"
#include "weighted_kv_query.h"
#include "weighted_kv_full.h"
#include "weighted_region.h"
#include "weighted_source_adaptation.h"
#include "tide/parameter_bank.h"
namespace tide::device_online {
// A role may reshape/select a registered parameter, but never acquires a new
// parameter identity. The reverse mapping adds all role/user contributions.
struct WeightedBankRole {
  std::string parameter;
  at::Tensor bank;
  std::vector<int64_t> original_shape;
  int64_t offset=0;
};
class WeightedBanks {
 public:
  WeightedBanks(const std::map<std::string,at::Tensor>&,std::string prefix,at::Device,
                std::shared_ptr<ParameterBankPool> pool=nullptr);
  // One binding group receives the owner's current remaining allowance. Each
  // role consumes it before the next role can allocate, even for tied parameters.
  void limit_bytes(int64_t available);
  at::Tensor bind(const std::string& role,const std::string& parameter,
                  const std::vector<int64_t>& shape,int64_t offset=0);
  at::Tensor direct(const std::string& role,const std::string& parameter);
  const std::map<std::string,WeightedBankRole>& roles()const{return roles_;}
  const std::map<std::string,at::Tensor>& parameters()const{return parameters_;}
  at::Device device()const{return device_;}
 private:
  const std::map<std::string,at::Tensor>& parameters_;
  std::string prefix_;at::Device device_;
  std::map<std::string,WeightedBankRole> roles_;
  std::shared_ptr<ParameterBankPool> pool_;
  int64_t available_=0;
};
WeightedDescribeParameters bind_weighted_describe(WeightedBanks&,int64_t width,int64_t summary,const weighted::RegionalConfig&,int64_t available);
WeightedRecurrentParameters bind_weighted_recurrent(WeightedBanks&,const weighted::RecurrentConfig&,int64_t available);
WeightedRegionParameters bind_weighted_region(WeightedBanks&,const weighted::RegionalConfig&,int64_t rows,int64_t available);
WeightedKVParameters bind_weighted_kv_update(WeightedBanks&,int64_t available);
WeightedKVQueryParameters bind_weighted_kv_query(WeightedBanks&,int64_t available);
WeightedKVFullParameters bind_weighted_kv_full(WeightedBanks&,int64_t available);
}
