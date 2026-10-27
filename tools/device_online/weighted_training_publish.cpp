#include "weighted_training_publish.h"
#include "peer_exchange.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <limits>
namespace tide::device_online {
namespace {
auto grouped(const std::vector<ParameterWrite>& writes,const at::Tensor& masters){
  std::map<c10::DeviceIndex,std::vector<ParameterWrite>> groups;
  for(const auto& w:writes){const auto device=w.destination.values.device();
    if(device.type()!=masters.device().type()||device.index()<0)throw std::invalid_argument("weighted publication backend differs");
    groups[device.index()].push_back(w);}
  return groups;
}
}
struct WeightedTrainingPublication::Impl {
  std::vector<std::unique_ptr<PeerExchange>> packets;
  std::vector<std::unique_ptr<DeviceProgram>> programs; // Destroy before packets.
  std::vector<at::Device> devices;
  uint32_t timeout;int64_t transferred=0;
};
int64_t WeightedTrainingPublication::required_bytes(const std::vector<ParameterWrite>& writes,const at::Tensor& masters){
  const auto groups=grouped(writes,masters);long double required=64.L*(writes.size()+groups.size());
  for(const auto& [id,group]:groups)if(id!=masters.device().index())required+=masters.nbytes()+4+1024;
  if(required>std::numeric_limits<int64_t>::max())throw std::invalid_argument("weighted publication extent overflow");
  return static_cast<int64_t>(required);
}
WeightedTrainingPublication::WeightedTrainingPublication(DeviceProgram& coordinator,
    const std::vector<ParameterWrite>& writes,const at::Tensor& masters,const at::Tensor& error,
    int64_t budget,int64_t workspace,uint32_t timeout):impl_(std::make_unique<Impl>()){
  auto& s=*impl_;s.timeout=timeout;
  const auto groups=grouped(writes,masters);const auto required=required_bytes(writes,masters);
  if(budget<1||required>budget)throw std::invalid_argument("weighted placed publication exceeds optimizer tensor budget before allocation");
  // The canonical optimizer already checks every owner's FP32/FP16 storage
  // representability before committing. All physical banks use that same
  // decision and rounded values; no independent optimizer runs on a shard.
  for(const auto& [id,group]:groups){
    if(id==masters.device().index()){append_owner_parameter_publish(coordinator,group,masters,error,budget);continue;}
    at::Device device(masters.device().type(),id);s.devices.push_back(device);
    auto program=std::make_unique<DeviceProgram>(device);program->limit_workspace(workspace);
    auto values=at::zeros(masters.sizes(),masters.options().device(device));
    auto decision=at::zeros(error.sizes(),error.options().device(device));
    auto packet=std::make_unique<PeerExchange>(PeerExchange::Fields{{error,decision},{masters,values}},budget,timeout);
    s.transferred+=packet->packet_bytes();packet->append_send(coordinator);packet->append_receive(*program);
    append_owner_parameter_publish(*program,group,values,decision,budget);program->finish();
    s.packets.push_back(std::move(packet));s.programs.push_back(std::move(program));
  }
}
WeightedTrainingPublication::~WeightedTrainingPublication()=default;
void WeightedTrainingPublication::synchronize_inputs(){
  for(auto device:impl_->devices)c10::impl::VirtualGuardImpl(device.type()).synchronizeDevice(device.index());
}
void WeightedTrainingPublication::submit(){for(auto& p:impl_->programs)p->submit();}
void WeightedTrainingPublication::wait(){std::exception_ptr error;
  for(auto& p:impl_->programs)try{p->wait(impl_->timeout);}catch(...){if(!error)error=std::current_exception();}
  if(error)std::rethrow_exception(error);
}
void WeightedTrainingPublication::close(){for(auto& p:impl_->programs)p->close();for(auto& x:impl_->packets)x->close();}
int64_t WeightedTrainingPublication::transferred_bytes()const{return impl_->transferred;}
}
