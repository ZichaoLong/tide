#include "state_owner.h"

namespace tide::device_online {
// Preserve full-profile entry points and class layouts. Compact owners use the
// explicit kernel-view overloads; original single-device clients keep their ABI.
void append_read(CannProgram& p,const ContentProfile& s,const ReadyBatch& r,const ContentBatch& c,
    const ContentState& old,const at::Tensor& a,const at::Tensor& e,int64_t ticks,bool vectorized,const at::Tensor& proposals) {
  append_read(p,StateKernelProfile(s),r,c,old,a,e,ticks,vectorized,proposals);
}
ContentUpdate append_content_state(CannProgram& p,const ContentProfile& s,const ReadyBatch& r,const ContentBatch& c,
    const SelectionProposal& selected,const ContentState& old,const at::Tensor& a,const at::Tensor& stage,
    const at::Tensor& count,const at::Tensor& e,const ContentLimits& limits,const at::Tensor& proposals) {
  return append_content_state(p,StateKernelProfile(s),r,c,selected,old,a,stage,count,e,limits,proposals);
}
long double PackedEventAttention::minimum_bytes(const ContentProfile& p,const Continuation& q,const ContentLimits& l) {
  return minimum_bytes(StateKernelProfile(p),q,l);
}
PackedEventAttention::PackedEventAttention(const ContentProfile& p,const Continuation& q,at::Device d,const ContentLimits& l,int64_t budget)
    :PackedEventAttention(StateKernelProfile(p),q,d,l,budget) {}
EventAttentionGroup::EventAttentionGroup(const ContentProfile& p,const Continuation& q,at::Device d,const ContentLimits& l,
    int64_t h,int64_t kh,int64_t chunk,int64_t keys):EventAttentionGroup(StateKernelProfile(p),q,d,l,h,kh,chunk,keys) {}
long double PackedFiberAttention::minimum_bytes(const ContentProfile& p,const Continuation& q,const ContentLimits& l) {
  return minimum_bytes(StateKernelProfile(p),q,l);
}
PackedFiberAttention::PackedFiberAttention(const ContentProfile& p,const Continuation& q,at::Device d,const ContentLimits& l,int64_t budget)
    :PackedFiberAttention(StateKernelProfile(p),q,d,l,budget) {}
FiberStage PackedFiberAttention::propose(CannProgram& p,const ContentProfile& s,const ReadyBatch& r,const ContentBatch& c,
    const ContentState& state,const at::Tensor& error) {return propose(p,StateKernelProfile(s),r,c,state,error);}
long double PackedFiberPool::reserved_bytes(const ContentProfile& p,int64_t rows,int64_t chunk) {
  return reserved_bytes(StateKernelProfile(p),rows,chunk);
}
PackedFiberPool::PackedFiberPool(const ContentProfile& p,at::Device d,int64_t rows,int64_t chunk)
    :PackedFiberPool(StateKernelProfile(p),d,rows,chunk) {}
} // namespace tide::device_online
