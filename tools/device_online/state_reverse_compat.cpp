#include "event_reverse.h"
#include "fiber_reverse.h"

namespace tide::device_online {
EventReverse prepare_event_reverse(DeviceProgram& p,const ReverseTape& t,const EventAttentionTape& a,
    const CacheCotangents& roots,const at::Tensor& error,int64_t budget) {
  return prepare_event_reverse(p,StateReverseView(t),a,roots,error,budget);
}
ReverseBatchPlan append_event_reverse(DeviceProgram& p,const ReverseTape& t,const EventAttentionTape& a,const EventReverse& reverse,
    const at::Tensor& range,StateVjp& state,const at::Tensor& parameters,const at::Tensor& connected,
    const at::Tensor& error,int64_t chunk,int64_t budget) {
  return append_event_reverse(p,StateReverseView(t),a,reverse,range,state,parameters,connected,error,chunk,budget);
}
FiberReverse prepare_fiber_reverse(DeviceProgram& p,const ReverseTape& t,const ReverseLinks& links,const FiberAttentionTape& a,
    const CacheCotangents& roots,const at::Tensor& error,int64_t budget) {
  return prepare_fiber_reverse(p,StateReverseView(t),links,a,roots,error,budget);
}
ReverseBatchPlan append_fiber_reverse(DeviceProgram& p,const ReverseTape& t,const ReverseLinks& links,const FiberAttentionTape& a,
    const FiberReverse& reverse,const at::Tensor& range,const StateVjp& state,const at::Tensor& messages,const at::Tensor& message_on,
    const at::Tensor& partials,const at::Tensor& parameters,const at::Tensor& parameter_on,const at::Tensor& error,int64_t chunk,int64_t budget) {
  return append_fiber_reverse(p,StateReverseView(t),links,a,reverse,range,state,messages,message_on,partials,parameters,parameter_on,error,chunk,budget);
}
} // namespace tide::device_online
