#include "device_backend.h"
#include "projection_shard.h"
#include "device_launch_tide_projection_shard_plan.h"
#include "device_launch_tide_full_vjp_reduce.h"
#include <ATen/core/grad_mode.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <algorithm>
#include <set>
#include <stdexcept>

namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
void merge_error(DeviceProgram& p,const at::Tensor& source,const at::Tensor& destination) {
  auto zero=at::zeros_like(source),equal=at::empty({1},source.options().dtype(at::kBool)),branch=at::empty_like(source);
  p.equal(source,zero,equal);p.cast_index(equal,branch);auto failed=p.label(),done=p.label();
  p.branch(branch,{failed,done});p.mark(failed);p.copy(destination,source);p.mark(done);
}
at::Tensor compute(DeviceProgram& p,const ProjectionBank& bank,const at::Tensor& map,int64_t parameters,
    const at::Tensor& global,const at::Tensor& input,const at::Tensor& cotangent,
    const at::Tensor& error,const ProjectionGradient& gradient) {
  const auto chunk=global.numel(),width=input.size(1),local=int64_t(bank.rows.size());
  const bool reverse=gradient.weights.defined(),half=bank.weights.scalar_type()==at::kHalf;
  auto l=global.options(),f=input.options();
  auto source=at::empty({chunk},l),param=at::empty_like(source),destination=at::empty_like(source),owners=at::empty_like(source);
  auto count=at::empty({1},l),branch=at::empty_like(error);
  auto connected=reverse?gradient.connected:at::empty({local},f.dtype(at::kBool));
  auto output=at::empty({2*chunk,width},f);p.zero(output);
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_projection_shard_plan)(1,stream,
    ptr(global),ptr(map),ptr(source),ptr(param),ptr(destination),ptr(owners),ptr(count),ptr(connected),ptr(branch),ptr(error),
    parameters,local,chunk,int64_t(reverse)),"pack actual projection rows for compact owner");},
    {global,map,source,param,destination,owners,count,connected,branch,error});
  auto body=p.label(),done=p.label();p.branch(branch,{done,body});p.mark(body);
  auto padded=at::zeros({chunk+1,width},f),x=at::empty({chunk,width},f),result=at::empty_like(x);
  p.copy(padded.narrow(0,0,chunk),input);p.index_select(padded,0,source,x);
  auto weights=at::empty({chunk,width,width},bank.weights.options());p.index_select(bank.weights,0,param,weights);
  if(!reverse) {
    auto bias=at::empty_like(x);p.index_select(bank.biases,0,param,bias);
    p.batch_matmul(x.reshape({chunk,1,width}),weights,result.reshape({chunk,1,width}));p.add(result,bias);
  } else {
    auto dy_pad=at::zeros_like(padded),dy=at::empty_like(x),w=half?at::empty(weights.sizes(),f):weights;
    p.copy(dy_pad.narrow(0,0,chunk),cotangent);p.index_select(dy_pad,0,source,dy);
    if(half)p.cast(weights,w);
    auto transposed=at::empty_like(w),dw=at::empty_like(w);
    p.permute(w,{0,2,1},transposed);p.batch_matmul(dy.reshape({chunk,1,width}),transposed,result.reshape({chunk,1,width}));
    p.batch_matmul(x.reshape({chunk,width,1}),dy.reshape({chunk,1,width}),dw);
    p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_full_vjp_reduce)(32,stream,
      ptr(owners),ptr(count),ptr(param),ptr(dw),ptr(dy),ptr(gradient.weights),ptr(gradient.biases),ptr(error),width,chunk),
      "ordered compact projection parameter reduction");},{owners,count,param,dw,dy,gradient.weights,gradient.biases,error});
  }
  p.index_copy(output,0,destination,result);p.mark(done);return output.narrow(0,0,chunk);
}
}
struct ProjectionStage::Impl {
  struct Shard {
    ProjectionBank bank;
    at::Tensor mapping,command,stop,result,error;
    ProjectionGradient gradient;
    std::unique_ptr<DeviceProgram> program;
    std::unique_ptr<PeerExchange> init,request,response;
  };
  std::vector<Shard> shards;
  int64_t parameters,chunk,width,tensor_budget,operator_budget;
  bool reverse,built=false,reset=false;
};
ProjectionStage::ProjectionStage(std::vector<ProjectionBank> banks,int64_t parameters,int64_t chunk,
    bool reverse,int64_t budget,int64_t workspace)
    :ProjectionStage(std::move(banks),parameters,chunk,reverse,budget,workspace,{}) {}
ProjectionStage::ProjectionStage(std::vector<ProjectionBank> banks,int64_t parameters,int64_t chunk,
    bool reverse,int64_t budget,int64_t workspace,const std::vector<ProjectionGradient>& reuse):impl_(std::make_unique<Impl>()) {
  if(at::GradMode::is_enabled()||banks.empty()||parameters<1||chunk<1||budget<1||workspace<1)
    throw std::invalid_argument("compact projections require bounded no-grad banks");
  auto& s=*impl_;s.parameters=parameters;s.chunk=chunk;s.reverse=reverse;s.tensor_budget=budget;s.operator_budget=workspace;
  const auto& first=banks.front().weights;
  if(!first.defined()||first.dim()!=3)throw std::invalid_argument("invalid compact projection extent");
  s.width=first.size(1);const auto dtype=first.scalar_type();
  if(s.width<1||(dtype!=at::kFloat&&dtype!=at::kHalf))throw std::invalid_argument("invalid compact projection dtype/width");
  std::vector<bool> seen(parameters);std::set<int> devices;
  // Numerical parameters are already owned by the forward/retained banks.
  // Charge new gradients and every owner's simultaneous chunk/packet buffers.
  long double bytes=0;
  for(const auto& bank:banks) {
    const auto n=int64_t(bank.rows.size());const auto d=bank.weights.device();
    if(n<1||d.type()!=tide::device_online::resident_device_type||d.index()<0||!devices.insert(d.index()).second
        ||!std::is_sorted(bank.rows.begin(),bank.rows.end()))throw std::invalid_argument("invalid compact projection owner");
    for(const auto& x:{bank.weights,bank.biases})
      if(!x.defined()||x.device()!=d||x.scalar_type()!=dtype||x.requires_grad()||!x.is_contiguous())
        throw std::invalid_argument("invalid compact projection bank");
    if(bank.weights.sizes()!=at::IntArrayRef{n+1,s.width,s.width}||bank.biases.sizes()!=at::IntArrayRef{n+1,s.width})
      throw std::invalid_argument("invalid compact projection bank shape");
    for(auto row:bank.rows){if(row<0||row>=parameters||seen[row])throw std::invalid_argument("invalid projection row partition");seen[row]=true;}
    bytes+=8.L*parameters+4096+chunk*((reverse?16.L:8.L)*s.width*s.width+256.L*s.width+128);
    if(reverse)bytes+=4.L*n*(s.width*static_cast<long double>(s.width)+s.width)+n;
  }
  if(std::find(seen.begin(),seen.end(),false)!=seen.end())throw std::invalid_argument("incomplete projection row partition");
  if(bytes>budget)throw std::invalid_argument("compact projection chunk exceeds tensor budget");
  if(!reuse.empty()) {
    if(!reverse||reuse.size()!=banks.size())throw std::invalid_argument("invalid reusable projection owner count");
    for(size_t i=0;i<banks.size();++i) {
      const auto& g=reuse[i];const auto& bank=banks[i];const int64_t n=bank.rows.size();
      if(g.rows!=bank.rows)throw std::invalid_argument("reusable projection row mapping changed");
      for(const auto& pair:std::vector<std::pair<at::Tensor,std::vector<int64_t>>>{{g.weights,{n,s.width,s.width}},
          {g.biases,{n,s.width}},{g.connected,{n}}}) {
        const auto& x=pair.first;
        if(!x.defined()||x.device()!=bank.weights.device()||x.sizes()!=at::IntArrayRef(pair.second)
            ||x.scalar_type()!=(pair.second.size()==1?at::kBool:at::kFloat)||!x.is_contiguous()||x.requires_grad()
            ||x.is_alias_of(bank.weights)||x.is_alias_of(bank.biases))
          throw std::invalid_argument("invalid reusable projection gradient bank");
      }
      if(g.weights.is_alias_of(g.biases)||g.weights.is_alias_of(g.connected)||g.biases.is_alias_of(g.connected))
        throw std::invalid_argument("reusable projection gradient banks alias");
    }
  }
  size_t ordinal=0;
  for(auto& bank:banks) {
    std::vector<int64_t> mapping(parameters,-1);for(size_t i=0;i<bank.rows.size();++i)mapping[bank.rows[i]]=i;
    Impl::Shard shard;shard.mapping=at::tensor(mapping,at::kLong).to(bank.weights.device());
    if(reverse) {
      const auto n=int64_t(bank.rows.size());const auto f=bank.weights.options().dtype(at::kFloat);
      shard.gradient=reuse.empty()?ProjectionGradient{bank.rows,at::empty({n,s.width,s.width},f),at::empty({n,s.width},f),at::empty({n},f.dtype(at::kBool))}:reuse[ordinal];
    }
    shard.bank=std::move(bank);s.shards.push_back(std::move(shard));++ordinal;
  }
}
ProjectionStage::~ProjectionStage()=default;
at::Tensor ProjectionStage::append(DeviceProgram& coordinator,const at::Tensor& rows,const at::Tensor& x,
    const at::Tensor& dy,const at::Tensor& error) {
  auto& s=*impl_;if(s.built)throw std::logic_error("compact projection call site already built");
  if(s.reverse&&!s.reset)throw std::logic_error("projection reverse requires a device reset boundary");
  const auto d=x.device();const auto dtype=s.reverse?at::kFloat:s.shards.front().bank.weights.scalar_type();
  if(x.sizes()!=at::IntArrayRef{s.chunk,s.width}||x.scalar_type()!=dtype||rows.sizes()!=at::IntArrayRef{s.chunk}
      ||rows.scalar_type()!=at::kLong||error.sizes()!=at::IntArrayRef{1}||error.scalar_type()!=at::kInt
      ||s.reverse!=dy.defined()||(s.reverse&&(dy.sizes()!=x.sizes()||dy.scalar_type()!=dtype)))
    throw std::invalid_argument("invalid compact projection request");
  for(const auto& t:{x,rows,error,dy})if(t.defined()&&(t.device()!=d||!t.is_contiguous()||t.requires_grad()))
    throw std::invalid_argument("invalid compact projection request ownership");
  s.built=true;
  for(auto& t:s.shards) {
    const auto peer=t.bank.weights.device();
    t.error=at::zeros_like(error);coordinator.copy(t.error,error);
    if(peer==d)continue;
    auto buffer=[&](const at::Tensor& v){return at::zeros(v.sizes(),v.options().device(peer));};
    t.command=at::zeros_like(error);t.stop=at::zeros_like(error);
    auto command=buffer(t.command),remote_rows=buffer(rows),remote_x=buffer(x),remote_error=buffer(error);
    auto remote_dy=s.reverse?buffer(dy):at::Tensor();
    PeerExchange::Fields fields{{t.command,command},{rows,remote_rows},{x,remote_x},{t.error,remote_error}};
    if(s.reverse)fields.emplace_back(dy,remote_dy);
    t.request=std::make_unique<PeerExchange>(fields,s.tensor_budget);
    if(!t.program){t.program=std::make_unique<DeviceProgram>(peer);t.program->limit_workspace(s.operator_budget);}
    auto& p=*t.program;
    auto head=p.label(),body=p.label(),done=p.label();auto again=at::zeros_like(command);
    p.mark(head);t.request->append_receive(p);p.branch(command,{done,body});p.mark(body);
    auto result=compute(p,t.bank,t.mapping,s.parameters,remote_rows,remote_x,remote_dy,remote_error,t.gradient);
    t.result=at::zeros_like(x);
    t.response=std::make_unique<PeerExchange>(PeerExchange::Fields{{result,t.result},{remote_error,t.error}},s.tensor_budget);
    t.response->append_send(p);p.branch(again,{head});p.mark(done);p.finish();
    coordinator.copy(t.command,at::ones_like(error));t.request->append_send(coordinator);
  }
  for(auto& t:s.shards)if(!t.program)t.result=compute(coordinator,t.bank,t.mapping,s.parameters,rows,x,dy,t.error,t.gradient);
  auto result=at::empty_like(x);coordinator.zero(result);
  for(auto& t:s.shards) {
    if(t.program)t.response->append_receive(coordinator);
    merge_error(coordinator,t.error,error);coordinator.add(result,t.result);
  }
  return result;
}
void ProjectionStage::append_reset(DeviceProgram& p,at::Device coordinator) {
  auto& s=*impl_;if(!s.reverse)return;
  if(s.reset||s.built)throw std::logic_error("projection reverse reset boundary repeated");s.reset=true;
  for(auto& shard:s.shards) {
    const auto d=shard.bank.weights.device();
    if(d!=coordinator) {
      shard.program=std::make_unique<DeviceProgram>(d);shard.program->limit_workspace(s.operator_budget);
      auto source=at::zeros({1},at::TensorOptions().device(coordinator).dtype(at::kInt));
      auto target=at::zeros({1},source.options().device(d));
      shard.init=std::make_unique<PeerExchange>(PeerExchange::Fields{{source,target}},64);
      shard.init->append_send(p);shard.init->append_receive(*shard.program);
    }
    auto& program=shard.program?*shard.program:p;
    for(const auto& v:{shard.gradient.weights,shard.gradient.biases,shard.gradient.connected})program.zero(v);
  }
}
void ProjectionStage::append_stop(DeviceProgram& p) {
  for(auto& s:impl_->shards)if(s.program){p.copy(s.command,s.stop);s.request->append_send(p);}
}
void ProjectionStage::synchronize_inputs() const {
  for(const auto& s:impl_->shards){const auto d=s.bank.weights.device();c10::impl::VirtualGuardImpl(d.type()).synchronizeDevice(d.index());}
}
void ProjectionStage::submit(){for(auto& s:impl_->shards)if(s.program)s.program->submit();}
void ProjectionStage::wait() {
  std::exception_ptr failure;for(auto& s:impl_->shards)if(s.program)try{s.program->wait();}catch(...){if(!failure)failure=std::current_exception();}
  if(failure)std::rethrow_exception(failure);
}
void ProjectionStage::close(){for(auto& s:impl_->shards){if(s.program)s.program->close();if(s.init)s.init->close();if(s.request)s.request->close();if(s.response)s.response->close();}}
std::vector<ProjectionGradient> ProjectionStage::gradients() const {
  std::vector<ProjectionGradient> out;if(impl_->reverse)for(const auto& s:impl_->shards)out.push_back(s.gradient);return out;
}
int64_t ProjectionStage::packet_bytes() const {int64_t n=0;for(const auto& s:impl_->shards){if(s.init)n+=s.init->packet_bytes();if(s.request)n+=s.request->packet_bytes()+s.response->packet_bytes();}return n;}
int64_t ProjectionStage::workspace_bytes() const {int64_t n=0;for(const auto& s:impl_->shards)if(s.program)n+=s.program->workspace_bytes();return n;}
int64_t ProjectionStage::retained_tensor_bytes() const {int64_t n=0;for(const auto& s:impl_->shards)if(s.program)n+=s.program->retained_tensor_bytes();return n;}
} // namespace tide::device_online
