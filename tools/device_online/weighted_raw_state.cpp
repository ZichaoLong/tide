#include "weighted_raw_state.h"
#include <ATen/core/grad_mode.h>
#include <limits>
#include <set>

namespace tide::device_online {
namespace {
constexpr auto maximum=std::numeric_limits<int64_t>::max();
void require(bool ok,const char* message){if(!ok)throw std::invalid_argument(message);}
void detached(const at::Tensor& x){
  require(x.defined()&&x.device().is_cpu()&&!x.requires_grad(),"raw state boundary requires detached CPU tensors");
}
bool payload(at::ScalarType type){return type==at::kFloat||type==at::kDouble||type==at::kHalf;}
std::vector<int64_t> shape(const WeightedRawField& f,int64_t n){auto s=f.shape;if(f.ragged)s.insert(s.begin(),n);return s;}
void field_values(const at::Tensor& x,at::ScalarType dtype,at::IntArrayRef sizes){
  detached(x);require(x.scalar_type()==dtype&&x.sizes()==sizes,"raw state field dtype/shape mismatch");
  require(at::isfinite(x).all().item<bool>(),"nonfinite raw state field");
}
int64_t checked_capacity(int64_t n){require(n>=0&&n<=maximum-4,"raw state cache capacity out of range");return n;}
}
void WeightedRawLayout::add(std::string name,std::vector<int64_t> shape,bool ragged){
  int64_t width=1;for(auto n:shape){require(n>0&&width<=maximum/n,"raw state field extent overflow");width*=n;}
  const auto count=ragged?capacity_:1;require(count==0||width<=(maximum-floating_width_)/count,"raw state layout overflow");
  fields_.push_back({std::move(name),std::move(shape),floating_width_,width,ragged});floating_width_+=width*count;
}
WeightedRawLayout::WeightedRawLayout(const weighted::KVConfig& c,int64_t capacity):family_("kv"),capacity_(checked_capacity(capacity)),horizon_(c.window){
  c.validate();require(capacity_>0&&c.width<maximum,"KV raw state needs positive bounded capacity");
  add("value",{c.width+1});add("keys",{c.width},true);add("values",{c.width},true);
  add("summaries",{c.summary_width},true);add("weights",{},true);
}
WeightedRawLayout::WeightedRawLayout(const weighted::RecurrentConfig& c,int64_t capacity):family_(c.memory_family),capacity_(checked_capacity(capacity)){
  c.validate();require(c.width<maximum,"raw state value extent overflow");
  require(c.basic()?capacity_==0:capacity_>0,"recurrent raw cache capacity/domain mismatch");
  horizon_=c.basic()?0:(c.conv_size-1)*c.conv_stride;
  add("value",{c.width+1});add("H",{c.matrices(),c.value_width,c.key_width});add("Z",{2,c.summary_width});add("W",{2});
  if(!c.basic())add("conv_values",{c.conv_width()},true);
}
const WeightedRawField& WeightedRawLayout::field(const std::string& name)const{
  for(const auto& f:fields_)if(f.name==name)return f;throw std::invalid_argument("unknown weighted raw field: "+name);
}
void validate_weighted_raw_buffers(const WeightedRawLayout& l,const WeightedRawState& s){
  require(!at::GradMode::is_enabled(),"raw state boundary requires explicit no-grad");
  require(s.floating.defined()&&s.integer.defined()&&s.floating.dim()==2&&s.floating.size(0)>0
    &&s.floating.size(1)==l.floating_width()&&payload(s.floating.scalar_type())&&s.floating.is_contiguous()
    &&!s.floating.requires_grad()&&s.integer.scalar_type()==at::kLong&&!s.integer.requires_grad()
    &&s.integer.device()==s.floating.device()&&s.integer.is_contiguous()
    &&s.integer.sizes()==at::IntArrayRef({s.floating.size(0),l.integer_width()}),"invalid weighted raw buffers");
}
WeightedRawState encode_weighted_raw_state(const WeightedRawLayout& l,const std::vector<State>& states,int64_t budget){
  require(!at::GradMode::is_enabled()&&!states.empty(),"raw state import requires explicit no-grad and nonempty rows");
  detached(states[0].value);const auto dtype=states[0].value.scalar_type();require(payload(dtype),"unsupported raw state payload dtype");
  const long double bytes=states.size()*(static_cast<long double>(l.floating_width())*states[0].value.element_size()+8.L*l.integer_width());
  require(budget>0&&bytes<=budget,"raw state tensor budget exceeded");
  WeightedRawState out{at::zeros({int64_t(states.size()),l.floating_width()},states[0].value.options()),
    at::zeros({int64_t(states.size()),l.integer_width()},at::kLong)};
  for(size_t row=0;row<states.size();++row){const auto& s=states[row];
    require(s.last_time>=-1&&s.observations>=0,"invalid raw state clock/count");
    int64_t n=0,time=0;std::set<std::string> expected;
    const std::string times=l.kv()?"times":"conv_times";
    if(l.kv()||l.convolution()){
      const auto& t=s.slots.at(times);detached(t);require(t.dim()==1&&t.scalar_type()==at::kLong,"raw cache times must be int64 vector");
      n=t.numel();require(n<=l.capacity()&&n<=s.observations,"raw cache capacity/count exceeded");
      require((t>=0).all().item<bool>()&&(t<=s.last_time).all().item<bool>()
        &&(n<2||(t.slice(0,1)>t.slice(0,0,n-1)).all().item<bool>()),"invalid raw cache timestamps");
      require(l.kv()?(!l.horizon()||n<=l.horizon()):n<=l.horizon()+1,"raw cache exceeds mathematical bound");
      out.integer[row].slice(0,4,4+n).copy_(t);expected.insert(times);
    }
    if(!l.kv()){
      const auto& t=s.slots.at("time");detached(t);require(t.dim()==0&&t.scalar_type()==at::kLong,"raw recurrent time must be int64 scalar");
      time=t.item<int64_t>();require(time==std::max<int64_t>(0,s.last_time)&&((s.observations==0)==(s.last_time==-1)),"raw recurrent clock/count mismatch");
      expected.insert("time");
    }
    out.integer[row][0]=s.last_time;out.integer[row][1]=s.observations;out.integer[row][2]=n;out.integer[row][3]=time;
    for(const auto& f:l.fields()){
      const auto& x=f.name=="value"?s.value:s.slots.at(f.name);field_values(x,dtype,shape(f,n));
      if(f.name=="value")require(x.eq(0).all().item<bool>(),"raw weighted adapter value must be zero");
      else expected.insert(f.name);
      if(f.name=="weights")require(x.ge(0).all().item<bool>()&&x.le(1).all().item<bool>(),"invalid raw KV weights");
      if(f.name=="W")require(x.ge(0).all().item<bool>(),"invalid raw summary weights");
      out.floating[row].slice(0,f.offset,f.offset+x.numel()).copy_(x.reshape({-1}));
    }
    require(s.slots.size()==expected.size(),"raw state has undeclared slots");
  }
  return out;
}
std::vector<State> decode_weighted_raw_state(const WeightedRawLayout& l,const WeightedRawState& buffer){
  validate_weighted_raw_buffers(l,buffer);detached(buffer.floating);std::vector<State> states;
  for(int64_t row=0;row<buffer.floating.size(0);++row){auto meta=buffer.integer[row];const auto n=meta[2].item<int64_t>();
    require(n>=0&&n<=l.capacity(),"invalid raw cache length");State s;s.last_time=meta[0].item<int64_t>();s.observations=meta[1].item<int64_t>();
    if(l.kv()||l.convolution())s.slots[l.kv()?"times":"conv_times"]=meta.slice(0,4,4+n).clone();
    if(!l.kv())s.slots["time"]=meta[3].clone();else require(meta[3].item<int64_t>()==0,"invalid reserved KV time");
    for(const auto& f:l.fields()){
      auto value=buffer.floating[row].slice(0,f.offset,f.offset+f.width*(f.ragged?n:1)).reshape(shape(f,n)).clone();
      if(f.name=="value")s.value=value;else s.slots.emplace(f.name,value);
    }
    states.push_back(std::move(s));
  }
  // Validate decoded logical fields only. Unused padding is deliberately opaque.
  encode_weighted_raw_state(l,states,std::numeric_limits<int64_t>::max());return states;
}
WeightedRawState copy_weighted_raw_state(const WeightedRawState& s,at::Device device){
  require(!at::GradMode::is_enabled()&&s.floating.defined()&&s.integer.defined()
    &&!s.floating.requires_grad()&&!s.integer.requires_grad(),"raw state transfer requires explicit detached boundary");
  return {s.floating.to(device).clone(),s.integer.to(device).clone()};
}
} // namespace tide::device_online
