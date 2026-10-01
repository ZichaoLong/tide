#include "sharded_optimizer.h"
#include <map>
#include <set>
#include <stdexcept>
namespace tide::device_online {
void validate_sharded_optimizer_owners(const std::vector<ParameterVjp>& partitions) {
  std::set<const void*> storage,identity;std::set<std::string> names;
  for(const auto& p:partitions)for(const auto& owner:p.owners) {
    if(!owner.value.defined()||!owner.value.device().is_cpu()||owner.value.numel()<1
        ||(owner.value.scalar_type()!=at::kFloat&&owner.value.scalar_type()!=at::kHalf)
        ||owner.aliases.empty()||owner.canonical!=owner.aliases.front()
        ||!identity.insert(owner.value.unsafeGetTensorImpl()).second
        ||!storage.insert(owner.value.storage().unsafeGetStorageImpl()).second)
      throw std::invalid_argument("sharded optimizer requires unique canonical owners without shared storage");
    for(const auto& name:owner.aliases)if(!names.insert(name).second)
      throw std::invalid_argument("sharded optimizer repeats a parameter alias");
  }
}
std::vector<std::unique_ptr<DeviceOptimizer>> make_sharded_optimizers(
    const std::vector<ParameterVjp>& partitions,DeviceOptimizerKind kind,std::vector<OptimizerGroup> groups,int64_t budget) {
  validate_sharded_optimizer_owners(partitions);ParameterRegistry registry;
  std::map<std::string,size_t> placement;
  for(size_t d=0;d<partitions.size();++d)for(const auto& owner:partitions[d].owners) {
    auto master=owner.value.to(at::kFloat);
    for(const auto& name:owner.aliases)registry.add(name,master);
    placement.emplace(owner.canonical,d);
  }
  std::unique_ptr<NamedOptimizer> validator;
  if(kind==DeviceOptimizerKind::sgd)validator=std::make_unique<SGD>(registry,std::move(groups));
  else if(kind==DeviceOptimizerKind::adamw)validator=std::make_unique<AdamW>(registry,std::move(groups));
  else throw std::invalid_argument("unknown sharded optimizer kind");
  std::vector<std::vector<OptimizerGroup>> local(partitions.size(),validator->groups());
  for(size_t d=0;d<partitions.size();++d)for(auto& group:local[d]) {
    std::vector<std::string> selected;
    for(const auto& name:group.parameters)if(placement.at(name)==d)selected.push_back(name);
    group.parameters=std::move(selected);
  }
  std::vector<std::unique_ptr<DeviceOptimizer>> out;
  for(size_t d=0;d<partitions.size();++d)
    out.push_back(std::make_unique<DeviceOptimizer>(partitions[d],kind,std::move(local[d]),budget));
  return out;
}
} // namespace tide::device_online
