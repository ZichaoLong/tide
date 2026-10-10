#include "sharded_full.h"
#include "content_profile.h"
#include "full_shard_pack.h"
#include "remote_full.h"
#include <c10/core/impl/VirtualGuardImpl.h>
#include <array>
#include <numeric>

namespace tide::device_online {
namespace {
struct Specification {
  std::vector<Node> nodes;
  std::vector<NodeWeights> weights;
  std::vector<int64_t> mapping,kinds,lh;
  std::array<long double,3> minimum;
};
std::vector<Specification> specifications(const ContentProfile& p,const FullPlacement& placement,int64_t capacity) {
  validate_full_placement(placement,p.graph.nodes.size(),placement.devices.front());
  std::vector<Specification> out(placement.devices.size());
  for(auto& s:out)s.mapping.assign(p.graph.nodes.size(),-1);
  for(size_t n=0;n<p.graph.nodes.size();++n) {
    auto& s=out[placement.owners[n]];const auto& node=p.graph.nodes[n];
    s.mapping[n]=s.nodes.size();s.nodes.push_back(node);s.weights.push_back(p.model.nodes[n]);
    s.kinds.push_back(!node.identity&&node.full=="tanh");s.lh.push_back(node.identity?0:lh_full_kind(node.full));
  }
  for(auto& s:out)s.minimum={PackedFull::minimum_bytes(s.kinds,p.width,p.dtype),
    PackedLhFull::minimum_bytes(s.lh,p.width,capacity,p.dtype),
    PackedSwiGluFull::minimum_bytes(s.nodes,p.width,p.dtype,capacity)};
  return out;
}
long double common_bytes(const ContentProfile& p,const FullPlacement& placement,int64_t capacity) {
  // Packing/gather/scratch, both packet endpoints, descriptors and owner maps.
  return placement.devices.size()*(64.L*capacity*(5.L*p.width+32)+32.L*p.graph.nodes.size()+4096);
}
void merge_error(DeviceProgram& p,const at::Tensor& source,const at::Tensor& destination) {
  auto zero=at::zeros_like(source),equal=at::empty({1},source.options().dtype(at::kBool)),branch=at::empty_like(source);
  p.equal(source,zero,equal);p.cast_index(equal,branch);auto failed=p.label(),done=p.label();
  p.branch(branch,{failed,done});p.mark(failed);p.copy(destination,source);p.mark(done);
}
}
struct ShardedFull::Impl {
  struct Shard {
    at::Device device;
    at::Tensor mapping,work,chunks;
    int64_t nodes,parameter_bytes=0;
    std::vector<int64_t> global_nodes;
    std::unique_ptr<PackedFull> full;
    std::unique_ptr<PackedLhFull> lh;
    std::unique_ptr<PackedSwiGluFull> swiglu;
    std::unique_ptr<RemoteFull> remote;
  };
  at::Device coordinator;
  int64_t reserved=0;
  at::Tensor chunks;
  std::vector<Shard> shards;
  explicit Impl(at::Device d):coordinator(d){}
};
long double ShardedFull::minimum_bytes(const ContentProfile& p,const FullPlacement& placement,int64_t capacity) {
  long double result=common_bytes(p,placement,capacity);
  for(const auto& s:specifications(p,placement,capacity))for(const auto x:s.minimum)result+=x;
  return result;
}
ShardedFull::ShardedFull(const ContentProfile& p,FullPlacement placement,at::Device coordinator,
    int64_t capacity,int64_t max_rows,int64_t budget):impl_(std::make_unique<Impl>(coordinator)) {
  validate_full_placement(placement,p.graph.nodes.size(),coordinator);
  const auto specs=specifications(p,placement,capacity);const auto common=common_bytes(p,placement,capacity);
  long double minimum=common;int64_t active=0;
  for(const auto& s:specs)for(const auto x:s.minimum){minimum+=x;active+=x>0;}
  if(capacity<1||max_rows<1||minimum>budget)throw std::invalid_argument("Full shards exceed tensor budget");
  const auto extra=(budget-minimum)/std::max<int64_t>(1,active);
  impl_->reserved=int64_t(common);auto longs=at::TensorOptions().device(coordinator).dtype(at::kLong);
  impl_->chunks=at::zeros({1},longs);
  for(size_t i=0;i<specs.size();++i) {
    const auto& s=specs[i];Impl::Shard shard{placement.devices[i],at::tensor(s.mapping,at::kLong).to(coordinator),
      at::zeros({2},longs),at::zeros({1},longs),int64_t(s.nodes.size())};
    for(size_t n=0;n<s.mapping.size();++n)if(s.mapping[n]>=0)shard.global_nodes.push_back(n);
    std::vector<Tensor> weights,biases,norm_weights,norm_biases;
    for(size_t n=0;n<s.nodes.size();++n) {
      const auto& w=s.weights[n];const auto kind=s.lh[n];weights.push_back(w.weight);biases.push_back(w.bias);
      norm_weights.push_back(kind&&(kind-1)%3?w.extra.at("lh_norm_weight"):at::ones_like(w.bias));
      norm_biases.push_back(kind&&(kind-1)%3==2?w.extra.at("lh_norm_bias"):at::zeros_like(w.bias));
    }
    auto allowance=[&](size_t n){return int64_t(s.minimum[n]+extra);};
    shard.full=std::make_unique<PackedFull>(s.kinds,at::stack(weights),at::stack(biases),shard.device,max_rows,allowance(0));
    impl_->reserved+=shard.full->reserved_bytes();
    if(s.minimum[1]>0) {
      shard.lh=std::make_unique<PackedLhFull>(s.lh,at::stack(norm_weights),at::stack(norm_biases),shard.device,capacity,max_rows,allowance(1));
      impl_->reserved+=shard.lh->reserved_bytes();
    }
    if(s.minimum[2]>0) {
      shard.swiglu=std::make_unique<PackedSwiGluFull>(s.nodes,s.weights,p.width,p.dtype,shard.device,capacity,max_rows,allowance(2));
      impl_->reserved+=shard.swiglu->reserved_bytes();
    }
    FullExtraTape t;if(shard.lh)shard.lh->tape(t);if(shard.swiglu)shard.swiglu->tape(t);
    for(const auto& x:{shard.full->weights(),shard.full->biases(),t.lh_weights,t.lh_biases,t.gate,t.up,t.down})
      if(x.defined())shard.parameter_bytes+=x.nbytes();
    if(shard.device==coordinator)shard.chunks=shard.full->chunks();
    impl_->shards.push_back(std::move(shard));
  }
  if(impl_->reserved>budget)throw std::logic_error("Full shard allocations exceeded admission");
}
ShardedFull::~ShardedFull()=default;
ActionBatch ShardedFull::append_stage(DeviceProgram& p,const ActionBatch& actions,const at::Tensor& content,
    const at::Tensor& comparison,const at::Tensor& error,int64_t operator_budget) {
  struct Stage {FullShardBatch packed;at::Tensor error;ActionBatch result;};
  std::vector<Stage> stages;
  auto output=at::zeros({actions.values.size(0)*2,actions.values.size(1)},actions.values.options());
  p.copy(output.narrow(0,0,actions.values.size(0)),actions.values);
  for(auto& s:impl_->shards) {
    auto packed=append_full_shard_pack(p,actions,content,comparison,s.mapping,s.nodes,s.work,error);
    auto local_error=at::zeros_like(error);p.copy(local_error,error);stages.push_back({packed,local_error,{}});
  }
  // Submit all peer requests before local computation or receiving a response.
  // Each branch depends on this stage's actual nonempty packed owner subset.
  for(size_t i=0;i<stages.size();++i) {
    auto& s=impl_->shards[i];auto& t=stages[i];if(s.device==impl_->coordinator)continue;
    auto send=p.label(),done=p.label();p.branch(t.packed.branch,{done,send});p.mark(send);
    s.remote=std::make_unique<RemoteFull>(*s.full,s.lh.get(),s.swiglu.get(),operator_budget);
    t.result=s.remote->append_send_stage(p,t.packed.actions,t.packed.content,t.packed.comparison,t.error,s.chunks);p.mark(done);
  }
  for(size_t i=0;i<stages.size();++i) {
    auto& s=impl_->shards[i];auto& t=stages[i];if(s.device!=impl_->coordinator)continue;
    t.result=s.full->append_stage(p,t.packed.actions,t.packed.comparison,t.error);
    if(s.lh)t.result=s.lh->append_stage(p,t.result,t.packed.comparison,t.error,s.chunks);
    if(s.swiglu)t.result=s.swiglu->append_stage(p,t.result,t.packed.content,t.packed.comparison,t.error,s.chunks);
  }
  p.zero(impl_->chunks);
  for(size_t i=0;i<stages.size();++i) {
    auto& s=impl_->shards[i];auto& t=stages[i];auto receive=p.label(),done=p.label();
    p.branch(t.packed.branch,{done,receive});p.mark(receive);
    if(s.remote)s.remote->append_receive_stage(p);
    merge_error(p,t.error,error);p.index_copy(output,0,t.packed.destinations,t.result.values);p.mark(done);
    p.add(impl_->chunks,s.chunks);
  }
  return {actions.coordinates,output.narrow(0,0,actions.values.size(0)),actions.valid};
}
void ShardedFull::append_stop(DeviceProgram& p){for(auto& s:impl_->shards)if(s.remote)s.remote->append_stop(p);}
void ShardedFull::reset_window(){impl_->chunks.zero_();for(auto& s:impl_->shards){s.work.zero_();s.chunks.zero_();s.full->chunks().zero_();}}
void ShardedFull::synchronize_inputs() const {
  for(const auto& s:impl_->shards)c10::impl::VirtualGuardImpl(s.device.type()).synchronizeDevice(s.device.index());
}
void ShardedFull::submit(){for(auto& s:impl_->shards)if(s.remote)s.remote->submit();}
void ShardedFull::wait(){std::exception_ptr error;for(auto& s:impl_->shards)if(s.remote)try{s.remote->wait();}catch(...){if(!error)error=std::current_exception();}if(error)std::rethrow_exception(error);}
void ShardedFull::close(){for(auto& s:impl_->shards)if(s.remote)s.remote->close();}
int64_t ShardedFull::reserved_bytes() const{return impl_->reserved;}
int64_t ShardedFull::program_count() const {int64_t n=1;for(const auto& s:impl_->shards)n+=s.device!=impl_->coordinator;return n;}
int64_t ShardedFull::workspace_bytes() const {int64_t n=0;for(const auto& s:impl_->shards)if(s.remote)n+=s.remote->workspace_bytes();return n;}
int64_t ShardedFull::packet_bytes() const {int64_t n=0;for(const auto& s:impl_->shards)if(s.remote)n+=s.remote->packet_bytes();return n;}
int64_t ShardedFull::retained_tensor_bytes() const {int64_t n=0;for(const auto& s:impl_->shards)if(s.remote)n+=s.remote->retained_tensor_bytes();return n;}
const at::Tensor& ShardedFull::chunks() const{return impl_->chunks;}
std::vector<FullShardTape> ShardedFull::tapes(int64_t samples,int64_t width) const {
  std::vector<FullShardTape> out;
  for(const auto& s:impl_->shards) {
    FullTape t{{},{},{},s.full->kinds(),s.full->weights(),s.full->biases(),samples,width,s.full->has_tanh()};
    if(s.lh)s.lh->tape(t.extra);if(s.swiglu)s.swiglu->tape(t.extra);
    out.push_back({s.global_nodes,std::move(t)});
  }
  return out;
}
std::map<std::string,int64_t> ShardedFull::stats() const {
  std::map<std::string,int64_t> out{{"full_shards",int64_t(impl_->shards.size())},{"full_shard_parameter_bytes",0},{"full_shard_max_parameter_bytes",0},
    {"full_shard_selected_rows",0},{"full_shard_capacity_rows",0}};
  for(size_t i=0;i<impl_->shards.size();++i) {
    const auto& s=impl_->shards[i];const auto work=s.work.cpu();
    out["full_shard_parameter_bytes"]+=s.parameter_bytes;
    out["full_shard_max_parameter_bytes"]=std::max(out["full_shard_max_parameter_bytes"],s.parameter_bytes);
    out["full_shard_selected_rows"]+=work[0].item<int64_t>();out["full_shard_capacity_rows"]+=work[1].item<int64_t>();
    out["full_shard_"+std::to_string(i)+"_nodes"]=s.nodes;
    out["full_shard_"+std::to_string(i)+"_parameter_bytes"]=s.parameter_bytes;
    out["full_shard_"+std::to_string(i)+"_full_chunk_rows"]=s.full->chunk_rows();
    out["full_shard_"+std::to_string(i)+"_lh_chunk_rows"]=s.lh?s.lh->chunk_rows():0;
    out["full_shard_"+std::to_string(i)+"_swiglu_chunk_rows"]=s.swiglu?s.swiglu->chunk_rows():0;
  }
  return out;
}
} // namespace tide::device_online
