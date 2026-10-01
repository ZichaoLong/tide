#include "sharded_parameter_reduce.h"
#include "owner_gradient_packet.h"
#include "peer_exchange.h"
#include "sharded_optimizer.h"
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <map>
#include <limits>
#include <set>
#include <stdexcept>

namespace tide::device_online {
namespace {
void sticky(CannProgram& p,const at::Tensor& source,const at::Tensor& target) {
  auto zero=at::zeros_like(source),equal=at::empty({1},source.options().dtype(at::kBool)),branch=at::empty_like(source);
  p.equal(source,zero,equal);p.cast_index(equal,branch);auto bad=p.label(),done=p.label();
  p.branch(branch,{bad,done});p.mark(bad);p.copy(target,source);p.mark(done);
}
struct Group {
  std::vector<ParameterContribution> parts;
  int64_t elements=0,value_base=0,flag_base=0;
};
struct Reference {int64_t source,offset,connection;};
}
struct ShardedParameterReduce::Impl {
  std::vector<at::Device> devices;
  std::vector<ParameterVjp> outputs;
  std::vector<at::Tensor> errors;
  std::vector<std::unique_ptr<PeerExchange>> packets;
  std::vector<std::unique_ptr<CannProgram>> programs; // Destroy before packets.
  int64_t budget;
  bool finished=false,updated=false,published=false;
  std::vector<DeviceOptimizer*> update_owners;
  void transfer(size_t source,size_t target,PeerExchange::Fields fields,int64_t capacity=0) {
    auto packet=std::make_unique<PeerExchange>(std::move(fields),capacity?capacity:budget);
    packet->append_send(*programs[source]);packet->append_receive(*programs[target]);packets.push_back(std::move(packet));
  }
  void consensus() {
    // Pair order is global and deterministic: every sender can reach its wait
    // before the coordinator receives it, then all await the common decision.
    for(size_t source=1;source<devices.size();++source) {
      auto received=at::zeros_like(errors[0]);transfer(source,0,{{errors[source],received}});
      sticky(*programs[0],received,errors[0]);
    }
    for(size_t target=1;target<devices.size();++target)transfer(0,target,{{errors[0],errors[target]}});
  }
};
ShardedParameterReduce::ShardedParameterReduce(ShardedParameterSources source,std::vector<at::Device> devices,
    const at::Tensor& upstream,int64_t budget,int64_t workspace):impl_(std::make_unique<Impl>()) {
  if(at::GradMode::is_enabled()||devices.empty()||devices.size()>16||budget<1||workspace<1||!upstream.defined()
      ||upstream.device()!=devices[0]||upstream.scalar_type()!=at::kInt||upstream.sizes()!=at::IntArrayRef({1})
      ||!upstream.is_contiguous()||upstream.requires_grad())
    throw std::invalid_argument("canonical owner reduction requires bounded no-grad NPU buffers");
  std::map<c10::DeviceIndex,size_t> index;
  for(size_t d=0;d<devices.size();++d)if(devices[d].type()!=c10::DeviceType::PrivateUse1||devices[d].index()<0
      ||!index.emplace(devices[d].index(),d).second)throw std::invalid_argument("invalid canonical owner device list");
  const auto placement=place_parameter_owners(source,devices.size());const auto n=devices.size();
  std::vector<std::vector<Group>> groups(n,std::vector<Group>(n)); // destination,source
  std::vector<std::vector<Reference>> references(source.owners.size());
  std::vector<int64_t> elements(n,0),flags(n,0),outputs(n,0),counts(n,0);
  long double bytes=4096.L*n+64.L*source.owners.size();
  for(size_t i=0;i<source.owners.size();++i) {
    const auto target=placement[i];const auto size=source.owners[i].value.numel();++counts[target];
    if(size<1)throw std::invalid_argument("empty canonical parameter owner");
    if(!source.contributions[i].empty()) {
      if(size>std::numeric_limits<int64_t>::max()-outputs[target])throw std::invalid_argument("canonical owner extent overflow");
      outputs[target]+=size;
    }
    for(const auto& c:source.contributions[i]) {
      if(!c.values.defined()||!c.connected.defined()||c.values.device().type()!=c10::DeviceType::PrivateUse1
          ||!index.count(c.values.device().index())||c.values.numel()!=size||c.values.device()!=c.connected.device()
          ||c.values.scalar_type()!=at::kFloat||c.connected.scalar_type()!=at::kBool||c.connected.numel()!=1
          ||!c.values.is_contiguous()||!c.connected.is_contiguous()||c.values.requires_grad()||c.connected.requires_grad())
        throw std::invalid_argument("invalid cross-owner parameter contribution");
      const auto from=index.at(c.values.device().index());auto& group=groups[target][from];
      if(size>std::numeric_limits<int64_t>::max()-group.elements)throw std::invalid_argument("owner packet extent overflow");
      references[i].push_back({int64_t(from),group.elements,int64_t(group.parts.size())});group.elements+=size;group.parts.push_back(c);
      bytes+=(from==size_t(target)?4.L:8.L)*size+128;
    }
  }
  for(size_t d=0;d<n;++d) {
    bytes+=4.L*std::max<int64_t>(1,outputs[d])+std::max<int64_t>(1,counts[d]);
    for(size_t from=0;from<n;++from) {
      auto& group=groups[d][from];group.value_base=elements[d];group.flag_base=flags[d];
      if(group.elements>std::numeric_limits<int64_t>::max()-elements[d])throw std::invalid_argument("owner receive extent overflow");
      elements[d]+=group.elements;flags[d]+=group.parts.size();
    }
  }
  if(bytes>budget)throw std::invalid_argument("canonical owner packets/reduction exceed tensor budget");
  auto& s=*impl_;s.devices=std::move(devices);s.budget=budget;s.outputs.resize(n);
  std::vector<at::Tensor> incoming(n),incoming_on(n);
  for(size_t d=0;d<n;++d) {
    auto f=at::TensorOptions().device(s.devices[d]).dtype(at::kFloat);
    s.programs.push_back(std::make_unique<CannProgram>(s.devices[d]));s.programs.back()->limit_workspace(workspace);
    s.errors.push_back(at::zeros({1},f.dtype(at::kInt)));
    incoming[d]=at::zeros({std::max<int64_t>(1,elements[d])},f);incoming_on[d]=at::zeros({std::max<int64_t>(1,flags[d])},f.dtype(at::kBool));
    s.outputs[d].values=at::empty({std::max<int64_t>(1,outputs[d])},f);
    s.outputs[d].connected=at::empty({std::max<int64_t>(1,counts[d])},f.dtype(at::kBool));
  }
  s.programs[0]->copy(s.errors[0],upstream);
  for(size_t target=1;target<n;++target)s.transfer(0,target,{{s.errors[0],s.errors[target]}});
  // All programs append device pairs in the same order. In particular, no
  // all-send/all-wait cycle is possible even when aliases cross both ways.
  for(size_t from=0;from<n;++from)for(size_t target=0;target<n;++target) {
    const auto& group=groups[target][from];if(group.parts.empty())continue;
    auto received=incoming[target].narrow(0,group.value_base,group.elements);
    auto received_on=incoming_on[target].narrow(0,group.flag_base,group.parts.size());
    auto send=from==target?received:at::empty({group.elements},received.options().device(s.devices[from]));
    auto send_on=from==target?received_on:at::empty({int64_t(group.parts.size())},received_on.options().device(s.devices[from]));
    append_owner_gradient_pack(*s.programs[from],group.parts,send,send_on,s.errors[from],budget);
    if(from!=target) {
      auto status=at::zeros_like(s.errors[target]);s.transfer(from,target,{{send,received},{send_on,received_on},{s.errors[from],status}});
      sticky(*s.programs[target],status,s.errors[target]);
    }
  }
  for(size_t target=0;target<n;++target) {
    auto& out=s.outputs[target];std::vector<int64_t> table,refs,tiles{0};int64_t offset=0;
    for(size_t owner=0;owner<source.owners.size();++owner)if(placement[owner]==int64_t(target)) {
      const auto first=int64_t(refs.size()/2),size=source.owners[owner].value.numel();
      for(const auto& ref:references[owner]) {
        const auto& group=groups[target][ref.source];refs.insert(refs.end(),{group.value_base+ref.offset,group.flag_base+ref.connection});
      }
      const auto last=int64_t(refs.size()/2),at=last>first?offset:-1;
      out.owners.push_back(source.owners[owner]);out.offsets.push_back(at);table.insert(table.end(),{first,last,at,size});
      if(at>=0)offset+=size;tiles.push_back(tiles.back()+(at>=0?(size+255)/256:0));
    }
    append_owner_gradient_reduce(*s.programs[target],table,refs,tiles,incoming[target],incoming_on[target],out,s.errors[target],budget);
  }
  s.consensus();
}
ShardedParameterReduce::~ShardedParameterReduce()=default;
ShardedParameterReduce::ShardedParameterReduce(std::vector<ParameterVjp> gradient,const at::Tensor& upstream,
    int64_t budget,int64_t workspace):impl_(std::make_unique<Impl>()) {
  if(at::GradMode::is_enabled()||gradient.empty()||gradient.size()>16||budget<1||workspace<1)
    throw std::invalid_argument("canonical update requires bounded no-grad gradients");
  validate_sharded_optimizer_owners(gradient);std::set<int> indices;auto& s=*impl_;s.budget=budget;
  if(4096.L*gradient.size()>budget)throw std::invalid_argument("canonical update metadata exceeds tensor budget");
  for(const auto& g:gradient) {
    if(!g.values.defined()||g.values.device().type()!=c10::DeviceType::PrivateUse1||g.values.device().index()<0
        ||!indices.insert(g.values.device().index()).second||g.values.scalar_type()!=at::kFloat||g.values.dim()!=1
        ||!g.connected.defined()||g.connected.device()!=g.values.device()||g.connected.scalar_type()!=at::kBool
        ||g.connected.sizes()!=at::IntArrayRef({std::max<int64_t>(1,g.owners.size())})
        ||!g.values.is_contiguous()||!g.connected.is_contiguous()||g.values.requires_grad()||g.connected.requires_grad())
      throw std::invalid_argument("invalid completed canonical gradient partition");
    s.devices.push_back(g.values.device());
  }
  if(!upstream.defined()||upstream.device()!=s.devices[0]||upstream.scalar_type()!=at::kInt
      ||upstream.sizes()!=at::IntArrayRef({1})||!upstream.is_contiguous()||upstream.requires_grad())
    throw std::invalid_argument("invalid canonical update upstream status");
  s.outputs=std::move(gradient);
  for(auto d:s.devices) {
    s.programs.push_back(std::make_unique<CannProgram>(d));s.programs.back()->limit_workspace(workspace);
    s.errors.push_back(at::zeros({1},upstream.options().device(d)));
  }
  s.programs[0]->copy(s.errors[0],upstream);
  for(size_t d=1;d<s.devices.size();++d)s.transfer(0,d,{{s.errors[0],s.errors[d]}});
}
const std::vector<ParameterVjp>& ShardedParameterReduce::gradients() const{return impl_->outputs;}
const std::vector<at::Tensor>& ShardedParameterReduce::errors() const{return impl_->errors;}
void ShardedParameterReduce::append_step(const std::vector<DeviceOptimizer*>& optimizers) {
  auto& s=*impl_;if(s.finished||s.updated||optimizers.size()!=s.programs.size())throw std::logic_error("invalid canonical owner update construction");
  for(auto* p:optimizers)if(!p)throw std::invalid_argument("missing canonical owner optimizer");
  validate_sharded_optimizer_owners(s.outputs);
  for(size_t d=0;d<s.programs.size();++d)optimizers[d]->append_propose(*s.programs[d],s.outputs[d],s.errors[d]);
  s.consensus();
  for(size_t d=0;d<s.programs.size();++d)optimizers[d]->append_commit(*s.programs[d],s.outputs[d],s.errors[d]);
  s.updated=true;s.update_owners=optimizers;
}
void ShardedParameterReduce::append_publish(const ShardedParameterBanks& banks,const std::vector<DeviceOptimizer*>& optimizers,int64_t budget) {
  auto& s=*impl_;const auto n=s.devices.size();
  if(s.finished||!s.updated||s.published||optimizers!=s.update_owners)throw std::logic_error("invalid sharded publication construction");
  if(budget<1)throw std::invalid_argument("sharded publication requires a positive tensor budget");
  const auto destinations=sharded_parameter_destinations(banks);
  std::map<c10::DeviceIndex,size_t> index;for(size_t d=0;d<n;++d)index.emplace(s.devices[d].index(),d);
  struct PublishGroup {std::vector<ParameterContribution> parts;std::vector<ParameterWrite> writes;int64_t size=0,base=0;};
  std::vector<std::vector<PublishGroup>> groups(n,std::vector<PublishGroup>(n)); // source,target
  std::vector<at::Tensor> yes;for(const auto& g:s.outputs)yes.push_back(at::ones({1},g.connected.options()));
  long double bytes=4096.L*n;std::vector<int64_t> receive_size(n,0);
  for(size_t from=0;from<n;++from) {
    const auto& layout=s.outputs[from];
    for(size_t i=0;i<layout.owners.size();++i) {
      const auto offset=layout.offsets[i];if(offset<0)continue;const auto& owner=layout.owners[i];
      std::vector<std::vector<ParameterDestination>> targets(n);
      for(const auto& name:owner.aliases)if(auto found=destinations.find(name);found!=destinations.end()) {
        const auto& dest=found->second;
        if(dest.values.sizes()!=owner.value.sizes()||dest.payload_dtype!=owner.value.scalar_type()
            ||!index.count(dest.values.device().index()))throw std::invalid_argument("publication alias shape/dtype/device mismatch");
        targets[index.at(dest.values.device().index())].push_back(dest);
      }
      for(size_t to=0;to<n;++to)if(!targets[to].empty()) {
        auto& group=groups[from][to];const auto size=owner.value.numel();
        bytes+=(from==to?4.L:8.L)*size+64.L*targets[to].size()+128;
        if(budget<1||bytes>budget)throw std::invalid_argument("sharded publication exceeds packet/metadata tensor budget");
        for(const auto& dest:targets[to])group.writes.push_back({group.size,dest});
        group.parts.push_back({optimizers[from]->values().narrow(0,offset,size),yes[from]});group.size+=size;
      }
    }
  }
  if(budget<1||bytes>budget)throw std::invalid_argument("sharded publication exceeds tensor budget");
  for(size_t from=0;from<n;++from)for(size_t to=0;to<n;++to) {
    auto& group=groups[from][to];group.base=receive_size[to];receive_size[to]+=group.size;
  }
  std::vector<at::Tensor> incoming;
  for(size_t d=0;d<n;++d)incoming.push_back(at::empty({std::max<int64_t>(1,receive_size[d])},s.outputs[d].values.options()));
  std::vector<std::vector<ParameterWrite>> writes(n);
  // The same global pair order as gradient reduction; no host polling or
  // numerical readback is required to deliver and publish updated aliases.
  for(size_t from=0;from<n;++from)for(size_t to=0;to<n;++to) {
    const auto& group=groups[from][to];if(group.parts.empty())continue;
    auto received=incoming[to].narrow(0,group.base,group.size);
    auto send=from==to?received:at::empty({group.size},s.outputs[from].values.options());
    auto on=at::empty({int64_t(group.parts.size())},s.outputs[from].connected.options());
    append_owner_gradient_pack(*s.programs[from],group.parts,send,on,s.errors[from],budget);
    if(from!=to)s.transfer(from,to,{{send,received}},budget);
    for(auto write:group.writes){write.offset+=group.base;writes[to].push_back(std::move(write));}
  }
  for(size_t d=0;d<n;++d)append_owner_parameter_publish(*s.programs[d],writes[d],incoming[d],s.errors[d],budget);
  s.published=true;
}
void ShardedParameterReduce::finish(){auto& s=*impl_;if(s.finished)throw std::logic_error("owner reduction already finished");for(auto& p:s.programs)p->finish();s.finished=true;}
void ShardedParameterReduce::run() {
  auto& s=*impl_;if(!s.finished)throw std::logic_error("owner reduction not finished");
  for(auto d:s.devices)c10::impl::VirtualGuardImpl(d.type()).synchronizeDevice(d.index());
  for(auto& p:s.programs)p->submit();std::exception_ptr failure;
  for(auto& p:s.programs)try{p->wait();}catch(...){if(!failure)failure=std::current_exception();}
  if(failure)std::rethrow_exception(failure);
}
void ShardedParameterReduce::close(){for(auto& p:impl_->programs)p->close();for(auto& p:impl_->packets)p->close();}
int64_t ShardedParameterReduce::packet_bytes() const{int64_t bytes=0;for(const auto& p:impl_->packets)bytes+=p->packet_bytes();return bytes;}
} // namespace tide::device_online
