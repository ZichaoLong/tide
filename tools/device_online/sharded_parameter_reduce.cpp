#include "sharded_parameter_reduce.h"
#include "owner_stream.h"
#include "peer_exchange.h"
#include "sharded_optimizer.h"
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <map>
#include <limits>
#include <set>
#include <tuple>
#include <stdexcept>

namespace tide::device_online {
namespace {
void sticky(CannProgram& p,const at::Tensor& source,const at::Tensor& target) {
  auto zero=at::zeros_like(source),equal=at::empty({1},source.options().dtype(at::kBool)),branch=at::empty_like(source);
  p.equal(source,zero,equal);p.cast_index(equal,branch);auto bad=p.label(),done=p.label();
  p.branch(branch,{bad,done});p.mark(bad);p.copy(target,source);p.mark(done);
}

}
struct ShardedParameterReduce::Impl {
  std::vector<at::Device> devices;
  std::vector<ParameterVjp> outputs;
  std::vector<at::Tensor> errors;
  std::vector<std::unique_ptr<PeerExchange>> packets;
  std::vector<std::unique_ptr<CannProgram>> programs; // Destroy before packets.
  int64_t budget,stream_bytes=0,stream_chunks=0;
  void stream(size_t from,size_t to,const std::vector<OwnerStreamField>& fields,int64_t capacity,bool accumulate) {
    auto plan=append_owner_stream(*programs[from],*programs[to],fields,errors[from],errors[to],capacity,accumulate);
    if(plan.reserved_bytes>std::numeric_limits<int64_t>::max()-stream_bytes
        ||plan.iterations>std::numeric_limits<int64_t>::max()-stream_chunks)
      throw std::invalid_argument("canonical stream statistics overflow");
    stream_bytes+=plan.reserved_bytes;stream_chunks+=plan.iterations;
    if(plan.peer)packets.push_back(std::move(plan.peer));
  }
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
  // All owners consume ordinal k before k+1, irrespective of source device.
  // Only independent owners are grouped by pair; floating addition order stays
  // reverse-window then registry-alias, including when aliases cross cards.
  using Key=std::tuple<size_t,size_t,size_t>; // ordinal,source,destination
  std::map<Key,std::vector<size_t>> groups;
  std::vector<int64_t> outputs(n,0),counts(n,0),offsets(source.owners.size(),-1),slots(source.owners.size());
  long double bytes=4096.L*n+64.L*source.owners.size();
  for(size_t i=0;i<source.owners.size();++i) {
    const auto target=placement[i],size=source.owners[i].value.numel();slots[i]=counts[target]++;
    if(!source.contributions[i].empty()) {
      if(size>std::numeric_limits<int64_t>::max()-outputs[target])throw std::invalid_argument("canonical owner extent overflow");
      offsets[i]=outputs[target];outputs[target]+=size;
    }
    for(size_t k=0;k<source.contributions[i].size();++k) {
      const auto& c=source.contributions[i][k];
      if(!c.values.defined()||!c.connected.defined()||c.values.device().type()!=c10::DeviceType::PrivateUse1
          ||!index.count(c.values.device().index())||c.values.numel()!=size||c.values.device()!=c.connected.device()
          ||c.values.scalar_type()!=at::kFloat||c.connected.scalar_type()!=at::kBool||c.connected.numel()!=1
          ||!c.values.is_contiguous()||!c.connected.is_contiguous()||c.values.requires_grad()||c.connected.requires_grad())
        throw std::invalid_argument("invalid cross-owner parameter contribution");
      groups[{k,index.at(c.values.device().index()),size_t(target)}].push_back(i);
    }
  }
  for(size_t d=0;d<n;++d)bytes+=4.L*std::max<int64_t>(1,outputs[d])+std::max<int64_t>(1,counts[d]);
  for(const auto& [key,owners]:groups) {
    const auto [k,from,to]=key;bytes+=owner_stream_metadata_bytes(owners.size(),owners.size(),from!=to);
  }
  if(bytes+8.L*groups.size()>budget)throw std::invalid_argument("canonical owner output/stream metadata exceed tensor budget");
  const auto allowance=groups.empty()?0:(budget-static_cast<int64_t>(bytes))/int64_t(groups.size());
  auto& s=*impl_;s.devices=std::move(devices);s.budget=budget;s.outputs.resize(n);
  for(size_t d=0;d<n;++d) {
    auto f=at::TensorOptions().device(s.devices[d]).dtype(at::kFloat);
    s.programs.push_back(std::make_unique<CannProgram>(s.devices[d]));s.programs.back()->limit_workspace(workspace);
    s.errors.push_back(at::zeros({1},f.dtype(at::kInt)));
    auto& out=s.outputs[d];out.values=at::empty({std::max<int64_t>(1,outputs[d])},f);
    out.connected=at::empty({std::max<int64_t>(1,counts[d])},f.dtype(at::kBool));
    s.programs[d]->zero(out.values);s.programs[d]->zero(out.connected);
  }
  for(size_t i=0;i<source.owners.size();++i) {
    auto& out=s.outputs[placement[i]];out.owners.push_back(source.owners[i]);out.offsets.push_back(offsets[i]);
  }
  s.programs[0]->copy(s.errors[0],upstream);
  for(size_t target=1;target<n;++target)s.transfer(0,target,{{s.errors[0],s.errors[target]}});
  // Global order also prevents all-send/all-wait cycles. Each packet is reused
  // by a device loop, so storage does not contain every numerical contribution.
  for(const auto& [key,owners]:groups) {
    const auto [k,from,to]=key;std::vector<OwnerStreamField> fields;const auto& out=s.outputs[to];
    for(auto i:owners)fields.push_back({source.contributions[i][k],
      {{out.values.narrow(0,offsets[i],source.owners[i].value.numel()),at::kFloat}},out.connected.narrow(0,slots[i],1)});
    s.stream(from,to,fields,owner_stream_metadata_bytes(fields.size(),fields.size(),from!=to)+allowance,true);
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
  using Key=std::pair<size_t,size_t>;
  std::map<Key,std::vector<OwnerStreamField>> groups;
  std::vector<at::Tensor> yes;for(const auto& g:s.outputs)yes.push_back(at::ones({1},g.connected.options()));
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
      for(size_t to=0;to<n;++to)if(!targets[to].empty())groups[{from,to}].push_back({
        {optimizers[from]->values().narrow(0,offset,owner.value.numel()),yes[from]},std::move(targets[to]),{}});
    }
  }
  std::map<Key,int64_t> metadata;long double bytes=4096.L*n;
  for(const auto& [key,fields]:groups) {
    int64_t writes=0;for(const auto& f:fields)writes+=f.destinations.size();
    metadata[key]=owner_stream_metadata_bytes(fields.size(),writes,key.first!=key.second);bytes+=metadata[key];
  }
  if(bytes+8.L*groups.size()>budget)throw std::invalid_argument("sharded publication stream metadata exceeds tensor budget");
  const auto allowance=groups.empty()?0:(budget-static_cast<int64_t>(bytes))/int64_t(groups.size());
  // Same global pair order; only updated masters travel, with strided local
  // aliases written directly. No complete numerical send/receive staging bank.
  for(const auto& [key,fields]:groups)s.stream(key.first,key.second,fields,metadata[key]+allowance,false);
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
int64_t ShardedParameterReduce::stream_reserved_bytes() const{return impl_->stream_bytes;}
int64_t ShardedParameterReduce::stream_chunks() const{return impl_->stream_chunks;}
int64_t ShardedParameterReduce::packet_bytes() const{int64_t bytes=0;for(const auto& p:impl_->packets)bytes+=p->packet_bytes();return bytes;}
} // namespace tide::device_online
