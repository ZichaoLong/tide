#include "block_settle_session_internal.h"
#include "weighted_storage_budget.h"
#include "resident_program_admission.h"
#include <algorithm>
#include <set>
namespace tide::device_online {
BlockSettleSession::Impl::Storage BlockSettleSession::Impl::storage(const DeviceProgram* current,const TaskServices* services)const{
  std::set<const DeviceProgram*> unique;
  std::set<const DeviceProgram*> groups;Storage out;
  auto add=[&](const DeviceProgram* coordinator,const TaskServices* calls){
    if(!coordinator||!groups.insert(coordinator).second)return;
    unique.insert(coordinator);
    if(calls){for(const auto* p:calls->programs())unique.insert(p);}
    const auto own=coordinator->resources();out.coordinators=resident_bytes_add(out.coordinators,own.programs);
    const auto stats=calls?calls->statistics():TaskServiceStatistics{0,0,0,coordinator->workspace_bytes(),own.programs,own.streams,0,own.boundary_events};
    out.workspaces=resident_bytes_add(out.workspaces,stats.shared_workspace_bytes);
    auto& total=out.services;
    for(const auto member:{&TaskServiceStatistics::phases,&TaskServiceStatistics::worker_programs,&TaskServiceStatistics::programs,
        &TaskServiceStatistics::streams,&TaskServiceStatistics::notifications,&TaskServiceStatistics::boundary_events})
      total.*member=resident_bytes_add(total.*member,stats.*member);
    total.maximum_nesting=std::max(total.maximum_nesting,stats.maximum_nesting);
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
  out.tensors=programs.empty()?weighted_storage_bytes(owned):programs.front()->retained_tensor_bytes(owned,programs);
  out.services.shared_workspace_bytes=out.workspaces;
  return out;
}
void BlockSettleSession::Impl::budget(const DeviceProgram* current,const TaskServices* services)const{
  const auto used=storage(current,services);
  require_resident_bytes(resident_bytes_add(used.tensors,used.workspaces),limits.total_bytes,"total tensor/workspace");
}
void BlockSettleSession::Impl::finish_and_admit(DeviceProgram& current,int64_t before,int64_t limit,const char* partition,int64_t persistent,TaskServices* services)const{
  finish_resident_program(current,services,[&]{
    const auto used=storage(&current,services);
    require_resident_bytes(resident_bytes_add(persistent,used.tensors-before),limit,partition);
    require_resident_bytes(resident_bytes_add(used.tensors,used.workspaces),limits.total_bytes,"total tensor/workspace");
  });
}
BlockSettleStorageUsage BlockSettleSession::storage_usage()const{
  const auto& s=*impl_;s.check();const auto used=s.storage();return {used.tensors,used.workspaces,s.optimizer->tensor_bytes()};
}
std::map<std::string,int64_t> BlockSettleSession::resource_usage()const{
  const auto& s=*impl_;s.check();const auto used=s.storage();const auto& calls=used.services;
  return {{"tensor_bytes",used.tensors},{"workspace_bytes",used.workspaces},{"optimizer_bytes",s.optimizer->tensor_bytes()},
    {"retained_windows",static_cast<int64_t>(s.windows.size())},{"coordinator_programs",used.coordinators},
    {"programs",calls.programs},{"worker_programs",calls.worker_programs},{"streams",calls.streams},
    {"notifications",calls.notifications},{"boundary_events",calls.boundary_events},
    {"service_phases",calls.phases},{"maximum_service_nesting",calls.maximum_nesting}};
}
}
