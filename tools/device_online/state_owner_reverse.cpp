#include "state_owner_reverse.h"
#include "cann_api.h"
#include "aclrtlaunch_tide_state_reverse_flags.h"
#include <stdexcept>
namespace tide::device_online {
namespace {
uint8_t* ptr(const at::Tensor& x){return static_cast<uint8_t*>(x.data_ptr());}
StateReverseView view(const StateOwnerTape& t,const StateReversePacket& p) {
  auto s=t.state;s.metadata=p.event_meta;s.values=p.event_values;s.count=p.event_count;
  return {t.layout,std::move(s),p.fiber_meta,p.fiber_values,t.sources};
}
}
StateOwnerReverse::StateOwnerReverse(CannProgram& p,const StateOwnerTape& t,const StateReversePacket& packet,
    const std::vector<CacheCotangents>& roots,const std::vector<CacheGradient>& next,const at::Tensor& error,int64_t chunk,int64_t budget)
    :owner_(t),view_(view(t,packet)),links_(packet.links),total_{t.global_nodes,t.layout},chunk_(chunk),budget_(budget) {
  const auto groups=t.attention.size()+t.fiber.size();const auto n=t.layout.nodes,w=t.layout.width,b=t.state.samples;
  if((!roots.empty()&&roots.size()!=groups)||(!next.empty()&&next.size()!=groups)||chunk<1||budget<1
      ||32.L*b*n*(w+1)+16.L*packet.fiber_values.numel()+8.L*(t.layout.event_offsets.back()+t.layout.fiber_offsets.back())+8192>budget/2.L)
    throw std::invalid_argument("compact state reverse accumulator/root budget exceeded");
  const auto f=packet.event_values.options(),bits=f.dtype(at::kBool);
  auto zeros=[&](at::IntArrayRef shape,bool flag=false){auto x=at::empty(shape,flag?bits:f);p.zero(x);return x;};
  total_.decay=zeros({n,w});total_.retention=zeros({n});total_.decay_connected=zeros({n},true);total_.retention_connected=zeros({n},true);
  total_.read=zeros({n,w});total_.read_connected=zeros({n},true);
  decay_=zeros({b,n,w});retention_=zeros({b,n,w});decay_on_=zeros({b,n},true);retention_on_=zeros({b,n},true);
  messages_=zeros(packet.fiber_values.sizes());connected_=zeros({packet.fiber_values.size(0)},true);partials_=zeros(packet.fiber_values.sizes());
  if(!t.attention.empty()) {total_.attention=zeros({t.layout.event_offsets.back()});total_.attention_connected=zeros({n,4},true);}
  if(!t.fiber.empty()){total_.fiber=zeros({t.layout.fiber_offsets.back()});total_.fiber_connected=zeros({n,6},true);}
  const int64_t per=budget/8/std::max<size_t>(1,groups);
  for(size_t i=0;i<t.attention.size();++i) {
    auto seed=roots.empty()?CacheCotangents{}:roots[i];
    if(!next.empty())seed=append_cache_bridge(p,t.attention[i],seed,next[i],error,per);
    events_.push_back(prepare_event_reverse(p,view_,t.attention[i],seed,error,per));total_.cache.push_back(events_.back().cache);
  }
  for(size_t i=0;i<t.fiber.size();++i) {
    const auto index=t.attention.size()+i;auto seed=roots.empty()?CacheCotangents{}:roots[index];
    if(!next.empty())seed=append_fiber_cache_seed(p,t.fiber[i],seed,&next[index],error,per);
    fibers_.push_back(prepare_fiber_reverse(p,view_,links_,t.fiber[i],seed,error,per));total_.cache.push_back(fibers_.back().cache);
  }
}
StateOwnerVjp StateOwnerReverse::append_stage(CannProgram& p,const StateReverseStage& stage,const at::Tensor& error) {
  if(built_)throw std::logic_error("compact state reverse stage already constructed");built_=true;
  if(stage.score_gradient.defined()) {
    const auto read=append_state_owner_read(p,owner_,stage,error,budget_/8);
    p.add(total_.read,read.read);append_connection_union(p,read.read_connected,total_.read_connected,error);
  }
  // Each source atom belongs to one reverse stage. Return this stage's cache
  // contribution so the coordinator adds it to its direct Aggregate derivative.
  for(const auto& x:{messages_,connected_,partials_})p.zero(x);
  auto state=append_state_vjp(p,stage.tape,stage.cot,error,budget_/4);
  for(size_t i=0;i<events_.size();++i)
    record_reverse_plan(total_.statistics,append_event_reverse(p,view_,owner_.attention[i],events_[i],stage.range,state,
      total_.attention,total_.attention_connected,error,chunk_,budget_/4/events_.size()),"event");
  for(size_t i=0;i<fibers_.size();++i)
    record_reverse_plan(total_.statistics,append_fiber_reverse(p,view_,links_,owner_.fiber[i],fibers_[i],stage.range,state,
      messages_,connected_,partials_,total_.fiber,total_.fiber_connected,error,chunk_,budget_/4/fibers_.size()),"fiber");
  p.add(decay_,state.decay);p.add(retention_,state.retention_components);
  append_connection_union(p,state.decay_connected,decay_on_,error);append_connection_union(p,state.retention_connected,retention_on_,error);
  return {state,messages_,connected_,partials_};
}
void StateOwnerReverse::append_finish(CannProgram& p,const at::Tensor& error) {
  if(finished_)throw std::logic_error("compact state reverse already finished");finished_=true;
  p.sum(decay_,0,false,total_.decay);auto features=at::empty_like(total_.decay);
  p.sum(retention_,0,false,features);p.sum(features,1,false,total_.retention);
  const auto n=owner_.layout.nodes,b=owner_.state.samples;auto d=decay_on_,r=retention_on_,od=total_.decay_connected,orr=total_.retention_connected;
  p.kernel([=](void* stream){CannApi::check(ACLRT_LAUNCH_KERNEL(tide_state_reverse_flags)(1,stream,
    ptr(d),ptr(r),ptr(od),ptr(orr),ptr(error),b,n),"fold compact state parameter connectivity");},{d,r,od,orr,error});
}
} // namespace tide::device_online
