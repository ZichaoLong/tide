// Sample-independent graphs share one summed update. No reference execution,
// host gradient reduction or per-event host decisions enter this composition.
#include "weighted_training_internal.h"
#include "weighted_training_publish.h"
#include "peer_exchange.h"
#include "device_launch_tide_weighted_peer_equal.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <limits>
#include <set>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
auto options(const OptimizerGroup& g){return std::tie(g.parameters,g.lr,g.weight_decay,g.momentum,g.dampening,
  g.beta1,g.beta2,g.eps,g.nesterov,g.amsgrad,g.maximize);}
template<class State>void compatible(const State& a,const State& b){
  if(a.kind!=b.kind||a.generation!=b.generation||a.layout.offsets!=b.layout.offsets
      ||a.layout.owners.size()!=b.layout.owners.size())throw std::invalid_argument("weighted replicas differ in optimizer/layout/generation");
  const auto& x=a.optimizer->groups();const auto& y=b.optimizer->groups();
  if(x.size()!=y.size())throw std::invalid_argument("weighted replica optimizer groups differ");
  for(size_t j=0;j<x.size();++j)if(options(x[j])!=options(y[j]))throw std::invalid_argument("weighted replica optimizer options differ");
  for(size_t j=0;j<a.layout.owners.size();++j){const auto& x=a.layout.owners[j];const auto& y=b.layout.owners[j];
    if(x.canonical!=y.canonical||x.aliases!=y.aliases||x.value.sizes()!=y.value.sizes()||x.value.scalar_type()!=y.value.scalar_type())
      throw std::invalid_argument("weighted replica canonical owners differ");}
}
std::vector<at::Tensor> slots(const DeviceOptimizer& x){return {x.values(),x.first(),x.second(),x.maximum(),x.steps(),x.corrections()};}
void same(DeviceProgram& p,const at::Tensor& a,const at::Tensor& b,const at::Tensor& error){
  const int64_t bytes=a.nbytes();
  if(a.device()!=b.device()||a.scalar_type()!=b.scalar_type()||a.sizes()!=b.sizes())throw std::invalid_argument("weighted replica slot layout differs");
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_peer_equal)(1,stream,
    ptr(a),ptr(b),ptr(error),bytes),"weighted replica exact optimizer identity");},{a,b,error});
}
void sticky(DeviceProgram& p,const at::Tensor& source,const at::Tensor& target){
  auto yes=at::zeros({1},source.options().dtype(at::kBool)),branch=at::zeros_like(source);
  p.equal(source,at::zeros_like(source),yes);p.cast_index(yes,branch);
  auto bad=p.label(),done=p.label();p.branch(branch,{bad,done});p.mark(bad);p.copy(target,source);p.mark(done);
}
ResidentParameterGradient public_gradient(const ParameterVjp& p){
  ResidentParameterGradient out;out.offsets=p.offsets;out.values=p.values;out.connected=p.connected;
  for(const auto& x:p.owners){out.names.push_back(x.canonical);out.aliases.push_back(x.aliases);}return out;
}
}
WeightedPeerStep WeightedTrainingSession::step_replicas(const std::vector<WeightedTrainingSession*>& sessions,
    Index budget,Index timeout){
  training_detail::no_grad();
  if(sessions.size()<2||sessions.size()>16||budget<1||timeout<1||timeout>60000)
    throw std::invalid_argument("weighted peer step needs 2..16 replicas, positive bytes and bounded timeout");
  std::vector<Impl*> owners;std::set<c10::DeviceIndex> devices;
  for(auto* session:sessions){
    if(!session||!session->impl_)throw std::invalid_argument("missing weighted replica");
    auto& s=*session->impl_;s.check();
    if(!devices.insert(s.device.index()).second||!s.saved.empty()||(!s.gradients_ready&&!s.accumulated_batches)
        ||s.gradients_ready&&s.accumulated_batches)throw std::invalid_argument("weighted peer step requires distinct devices and completed gradients");
    if(s.generation==std::numeric_limits<Index>::max()||s.flow->impl_->generation==std::numeric_limits<Index>::max())
      throw std::overflow_error("weighted replica generation exhausted");
    if(!owners.empty()){
      const auto& a=*owners[0];compatible(a,s);
      if(a.flow->impl_->graph.identity!=s.flow->impl_->graph.identity)throw std::invalid_argument("weighted replica graphs differ");
      const auto x=a.flow->impl_->model.parameters(false),y=s.flow->impl_->model.parameters(false);
      if(x.alias_partitions()!=y.alias_partitions()||x.names()!=y.names())throw std::invalid_argument("weighted replica parameter aliases differ");
      for(const auto& p:x.owners())if(p.value.scalar_type()!=y.value(p.canonical).scalar_type()||!at::equal(p.value,y.value(p.canonical)))
        throw std::invalid_argument("weighted replica named parameters differ");
    }
    owners.push_back(&s);
  }
  const auto n=owners.size();auto& lead=*owners[0];
  // Conservative: all peer receives, exact slot copies, accumulated copies,
  // metadata and returned sums coexist. No allocation/search by trial OOM.
  long double reserve=4096.L*n;
  for(auto* s:owners){reserve+=4.L*(s->layout.values.nbytes()+s->layout.connected.nbytes());
    for(const auto& x:slots(*s->optimizer))reserve+=x.nbytes();
    reserve+=WeightedTrainingPublication::required_bytes(s->writes(),s->optimizer->values());}
  if(reserve>budget)throw std::invalid_argument("weighted peer communication tensor budget exceeded before allocation");
  std::vector<std::unique_ptr<PeerExchange>> packets;
  std::vector<std::unique_ptr<DeviceProgram>> programs; // Destruct before packets.
  std::vector<at::Tensor> errors;std::vector<ParameterVjp> sums;
  WeightedPeerStep out;out.step={false,0,lead.generation};out.reserved_tensor_bytes=static_cast<Index>(reserve);
  for(auto* s:owners){programs.push_back(std::make_unique<DeviceProgram>(s->device));programs.back()->limit_workspace(s->limits.forward.operator_bytes);
    errors.push_back(at::zeros({1},s->layout.values.options().dtype(at::kInt)));
    auto g=s->accumulated_batches?s->accumulated:s->gradient;
    if(s==owners[0]){g.values=at::zeros_like(g.values);g.connected=at::zeros_like(g.connected);
      const auto& source=s->accumulated_batches?s->accumulated:s->gradient;
      programs.back()->copy(g.values,source.values);programs.back()->copy(g.connected,source.connected);}
    sums.push_back(std::move(g));
  }
  auto transfer=[&](size_t from,size_t to,PeerExchange::Fields fields){
    auto packet=std::make_unique<PeerExchange>(std::move(fields),budget,static_cast<uint32_t>(timeout));
    out.transferred_bytes+=packet->packet_bytes();packet->append_send(*programs[from]);packet->append_receive(*programs[to]);packets.push_back(std::move(packet));
  };
  for(size_t j=1;j<n;++j){
    const auto a=slots(*lead.optimizer),b=slots(*owners[j]->optimizer);PeerExchange::Fields fields;std::vector<at::Tensor> received;
    for(size_t k=0;k<a.size();++k){
      if(a[k].sizes()!=b[k].sizes()||a[k].scalar_type()!=b[k].scalar_type())throw std::invalid_argument("weighted replica slots differ");
      received.push_back(at::zeros_like(a[k]));fields.push_back({b[k],received.back()});}
    auto part=sums[0];part.values=at::zeros_like(sums[0].values);part.connected=at::zeros_like(sums[0].connected);
    fields.push_back({sums[j].values,part.values});fields.push_back({sums[j].connected,part.connected});transfer(j,0,std::move(fields));
    for(size_t k=0;k<a.size();++k)same(*programs[0],a[k],received[k],errors[0]);
    sums[0]=append_parameter_accumulate(*programs[0],sums[0],part,errors[0],budget);
  }
  for(size_t j=1;j<n;++j){sums[j].values=at::zeros_like(sums[j].values);sums[j].connected=at::zeros_like(sums[j].connected);
    transfer(0,j,{{sums[0].values,sums[j].values},{sums[0].connected,sums[j].connected},{errors[0],errors[j]}});}
  for(size_t j=0;j<n;++j)owners[j]->optimizer->append_propose(*programs[j],sums[j],errors[j]);
  for(size_t j=1;j<n;++j){auto e=at::zeros_like(errors[0]);transfer(j,0,{{errors[j],e}});sticky(*programs[0],e,errors[0]);}
  for(size_t j=1;j<n;++j)transfer(0,j,{{errors[0],errors[j]}});
  std::vector<std::unique_ptr<WeightedTrainingPublication>> publications;
  for(size_t j=0;j<n;++j){auto& s=*owners[j];s.optimizer->append_commit(*programs[j],sums[j],errors[j]);
    publications.push_back(std::make_unique<WeightedTrainingPublication>(*programs[j],s.writes(),s.optimizer->values(),errors[j],
      s.limits.optimizer_bytes/4,s.limits.forward.operator_bytes,static_cast<uint32_t>(timeout)));
    out.transferred_bytes+=publications.back()->transferred_bytes();
    if(programs[j]->retained_tensor_bytes()>budget+s.limits.optimizer_bytes)
      throw std::invalid_argument("weighted peer programs exceed admitted tensor budget");
    programs[j]->finish();}
  try{
    for(auto& publication:publications)publication->synchronize_inputs();
    for(auto* s:owners)c10::impl::VirtualGuardImpl(s->device.type()).synchronizeDevice(s->device.index());
    for(auto& publication:publications)publication->submit();
    for(auto& p:programs)p->submit();for(auto& p:programs)p->wait(static_cast<int32_t>(timeout));
    for(auto& publication:publications)publication->wait();
    const auto code=errors[0].cpu().item<int>();
    for(auto& p:programs)p->close();for(auto& publication:publications)publication->close();for(auto& packet:packets)packet->close();
    if(code&&code!=20&&code!=21&&code!=25)throw std::runtime_error("weighted peer optimizer refused, error="+std::to_string(code));
    out.step={code==0,code,lead.generation+(code==0)};
    for(const auto& g:sums)out.parameters.push_back(public_gradient(g));
    if(!code)for(auto* s:owners){s->mirror_parameters();++s->generation;++s->flow->impl_->generation;s->flow->impl_->ran=false;s->discard();}
    return out;
  }catch(...){for(auto* s:owners)s->failed=true;throw;}
}
}
