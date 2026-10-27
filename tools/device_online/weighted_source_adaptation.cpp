#include "weighted_source_adaptation.h"
#include "weighted_aggregate_internal.h"
#include "device_launch_tide_weighted_source_guard.h"
namespace tide::device_online {
namespace {
using namespace aggregate_detail;
struct SourceShape {
  int64_t banks,hidden,d,rank;
  SourceShape(const WeightedSourceParameters& base,const WeightedSourceFactors& f,at::Device device){
    if(!base.a.defined()||base.a.dim()!=3||base.a.size(0)<1||base.a.size(1)<1||base.a.size(2)<1
        ||!f.a_left.defined()||f.a_left.dim()!=3||f.a_left.size(2)<1||at::GradMode::is_enabled()
        ||device.type()!=resident_device_type)throw std::invalid_argument("invalid resident source adaptation dimensions");
    validate_kernel_device(device);banks=base.a.size(0);hidden=base.a.size(1);d=base.a.size(2);rank=f.a_left.size(2);
    check(base.a,{banks,hidden,d},at::kFloat,device);check(base.b,{banks,d,hidden},at::kFloat,device);
    check(base.embedding,{banks,hidden},at::kFloat,device);
    check(f.a_left,{banks,hidden,rank},at::kFloat,device);check(f.a_right,{banks,rank,d},at::kFloat,device);
    check(f.b_left,{banks,d,rank},at::kFloat,device);check(f.b_right,{banks,rank,hidden},at::kFloat,device);
  }
};
at::Tensor guarded(DeviceProgram& p,const at::Tensor& x,const at::Tensor& roots,const at::Tensor& error,int64_t stage){
  auto result=at::zeros_like(x),errors=at::zeros({x.size(0)},error.options());
  const int64_t rows=x.size(0),width=x.numel()/rows;
  p.kernel([=](void* stream){check_device_launch(TIDE_LAUNCH_KERNEL(tide_weighted_source_guard)(1,stream,
    ptr(x),ptr(roots),ptr(result),ptr(errors),ptr(error),rows,width,stage),"weighted source finite/adjoint guard");},
    {x,roots,result,errors,error});status(p,errors,error);return result;
}
at::Tensor transpose(DeviceProgram& p,const at::Tensor& x){
  auto output=at::zeros({x.size(0),x.size(2),x.size(1)},x.options());p.permute(x,{0,2,1},output);return output;
}
}
WeightedSourceAdaptation append_weighted_source_adaptation(DeviceProgram& p,const WeightedSourceParameters& base,
    const WeightedSourceFactors& f,const at::Tensor& error,int64_t budget){
  if(!base.a.defined())throw std::invalid_argument("missing source base");const auto device=base.a.device();
  SourceShape s(base,f,device);check(error,{1},at::kInt,device);
  admit(4.L*s.banks*(6.L*s.hidden*s.d+s.hidden+2.L*s.rank*(s.hidden+s.d))+4.L*s.banks*9,budget);
  WeightedSourceAdaptation t;
  // Validate/copy actual factors once; neither reverse nor later optimizer
  // generations reread the input buffers of this saved invocation.
  t.saved={guarded(p,f.a_left,error,error,0),guarded(p,f.a_right,error,error,0),
    guarded(p,f.b_left,error,error,0),guarded(p,f.b_right,error,error,0)};
  auto a=guarded(p,base.a,error,error,0),b=guarded(p,base.b,error,error,0);
  t.effective={at::zeros_like(base.a),at::zeros_like(base.b),guarded(p,base.embedding,error,error,0)};
  p.batch_matmul(t.saved.a_left,t.saved.a_right,t.effective.a);p.add(t.effective.a,a);
  p.batch_matmul(t.saved.b_left,t.saved.b_right,t.effective.b);p.add(t.effective.b,b);
  // All configured parameters are finite, including currently unused banks.
  t.effective.a=guarded(p,t.effective.a,error,error,0);t.effective.b=guarded(p,t.effective.b,error,error,0);
  return t;
}
WeightedSourceAdaptationGradient append_weighted_source_adaptation_vjp(DeviceProgram& p,const WeightedSourceAdaptation& t,
    const WeightedSourceParameters& gradients,const at::Tensor& roots,const at::Tensor& error,int64_t budget){
  if(!t.effective.a.defined())throw std::invalid_argument("missing source adaptation tape");const auto device=t.effective.a.device();
  SourceShape s(t.effective,t.saved,device);check(error,{1},at::kInt,device);check(roots,{s.banks},at::kBool,device);
  check(gradients.a,{s.banks,s.hidden,s.d},at::kFloat,device);check(gradients.b,{s.banks,s.d,s.hidden},at::kFloat,device);
  check(gradients.embedding,{s.banks,s.hidden},at::kFloat,device);
  admit(4.L*s.banks*(2.L*s.hidden*s.d+s.hidden+8.L*s.rank*(s.hidden+s.d))+4.L*s.banks*11+s.banks,budget);
  WeightedSourceAdaptationGradient g;g.connected=at::empty_like(roots);p.copy(g.connected,roots);
  g.base={guarded(p,gradients.a,g.connected,error,1),guarded(p,gradients.b,g.connected,error,1),
    guarded(p,gradients.embedding,g.connected,error,1)};
  auto a_left=guarded(p,t.saved.a_left,g.connected,error,1),a_right=guarded(p,t.saved.a_right,g.connected,error,1);
  auto b_left=guarded(p,t.saved.b_left,g.connected,error,1),b_right=guarded(p,t.saved.b_right,g.connected,error,1);
  g.factors={at::zeros_like(a_left),at::zeros_like(a_right),at::zeros_like(b_left),at::zeros_like(b_right)};
  p.batch_matmul(g.base.a,transpose(p,a_right),g.factors.a_left);
  p.batch_matmul(transpose(p,a_left),g.base.a,g.factors.a_right);
  p.batch_matmul(g.base.b,transpose(p,b_right),g.factors.b_left);
  p.batch_matmul(transpose(p,b_left),g.base.b,g.factors.b_right);
  g.factors={guarded(p,g.factors.a_left,g.connected,error,1),guarded(p,g.factors.a_right,g.connected,error,1),
    guarded(p,g.factors.b_left,g.connected,error,1),guarded(p,g.factors.b_right,g.connected,error,1)};
  return g;
}
}
