#include "block_settle_session_internal.h"
#include "weighted_storage_budget.h"
#include <set>
namespace tide::device_online {
BlockSettleSession::Impl::Storage BlockSettleSession::Impl::storage(const DeviceProgram* current)const{
  std::set<const DeviceProgram*> unique;
  for(const auto& w:windows)unique.insert(w.program.get());
  if(reverse)unique.insert(reverse.get());if(current)unique.insert(current);
  const std::vector<const DeviceProgram*> programs(unique.begin(),unique.end());
  auto owned=optimizer->tensors();
  owned.insert(owned.end(),{boundary.node_present,boundary.region_times,boundary.input_counts,boundary.input_times});
  for(const auto& b:bindings){for(const auto& [name,x]:block_kv_local_parameters(b.parameters))owned.push_back(x);
    owned.insert(owned.end(),{b.parameters.message.read.region.context_owners,b.parameters.message.read.region.query_owners});}
  for(const auto& group:{state,bound})for(const auto& s:group){owned.push_back(s.floating);owned.push_back(s.integer);}
  Storage out{programs.empty()?weighted_storage_bytes(owned):programs.front()->retained_tensor_bytes(owned,programs),0};
  // These programs retain distinct vendor arenas even though submissions are
  // serialized. Count every live high-water mark, not one global maximum.
  for(const auto* p:programs)out.workspaces=resident_bytes_add(out.workspaces,p->workspace_bytes());
  return out;
}
void BlockSettleSession::Impl::budget(const DeviceProgram* current)const{
  const auto used=storage(current);
  require_resident_bytes(resident_bytes_add(used.tensors,used.workspaces),limits.total_bytes,"total tensor/workspace");
}
void BlockSettleSession::Impl::finish_and_admit(DeviceProgram& current,int64_t before,int64_t limit,const char* partition,int64_t persistent)const{
  current.finish(); // CUDA lowering may retain additional branch maps here.
  require_resident_bytes(resident_bytes_add(persistent,storage(&current).tensors-before),limit,partition);
  budget(&current);
}
BlockSettleStorageUsage BlockSettleSession::storage_usage()const{
  const auto& s=*impl_;s.check();const auto used=s.storage();return {used.tensors,used.workspaces,s.optimizer->tensor_bytes()};
}
}
