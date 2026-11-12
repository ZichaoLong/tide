#include "block_settle_session_internal.h"
#include "weighted_storage_budget.h"
#include <set>
namespace tide::device_online {
BlockSettleSession::Impl::Storage BlockSettleSession::Impl::storage(const DeviceProgram* current,const TaskServices* services)const{
  std::set<const DeviceProgram*> unique;
  std::set<const DeviceProgram*> groups;int64_t workspaces=0;
  auto add=[&](const DeviceProgram* coordinator,const TaskServices* calls){
    if(!coordinator||!groups.insert(coordinator).second)return;
    unique.insert(coordinator);
    if(calls){for(const auto* p:calls->programs())unique.insert(p);}
    workspaces=resident_bytes_add(workspaces,calls?calls->workspace_bytes():coordinator->workspace_bytes());
  };
  for(const auto& w:windows)add(w.program.get(),w.services.get());
  add(reverse.get(),reverse_services.get());add(current,services);
  const std::vector<const DeviceProgram*> programs(unique.begin(),unique.end());
  auto owned=optimizer->tensors();
  owned.insert(owned.end(),{boundary.node_present,boundary.region_times,boundary.input_counts,boundary.input_times});
  for(const auto& b:bindings){for(const auto& [name,x]:block_kv_local_parameters(b.parameters))owned.push_back(x);
    owned.insert(owned.end(),{b.parameters.message.read.region.context_owners,b.parameters.message.read.region.query_owners});}
  for(const auto& group:{state,bound})for(const auto& s:group){owned.push_back(s.floating);owned.push_back(s.integer);}
  // Each coordinator and its serialized numerical workers share one arena.
  // Different retained forward/reverse windows still own separate arenas.
  return {programs.empty()?weighted_storage_bytes(owned):programs.front()->retained_tensor_bytes(owned,programs),workspaces};
}
void BlockSettleSession::Impl::budget(const DeviceProgram* current,const TaskServices* services)const{
  const auto used=storage(current,services);
  require_resident_bytes(resident_bytes_add(used.tensors,used.workspaces),limits.total_bytes,"total tensor/workspace");
}
void BlockSettleSession::Impl::finish_and_admit(DeviceProgram& current,int64_t before,int64_t limit,const char* partition,int64_t persistent,TaskServices* services)const{
  if(services){services->append_stop();services->finish();}
  current.finish(); // CUDA lowering may retain additional branch maps here.
  require_resident_bytes(resident_bytes_add(persistent,storage(&current,services).tensors-before),limit,partition);
  budget(&current,services);
}
BlockSettleStorageUsage BlockSettleSession::storage_usage()const{
  const auto& s=*impl_;s.check();const auto used=s.storage();return {used.tensors,used.workspaces,s.optimizer->tensor_bytes()};
}
}
