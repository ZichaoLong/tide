#include "weighted_recurrent_internal.h"
namespace tide::device_online {
using namespace recurrent_detail;
WeightedRecurrentTape append_weighted_recurrent(DeviceProgram& p,const weighted::RecurrentConfig& c,WeightedRecurrentAction action,
    const WeightedRawState& raw,const at::Tensor& times,const at::Tensor& mass,const at::Tensor& content,
    const at::Tensor& owners,const at::Tensor& present,const WeightedRecurrentParameters& w,const at::Tensor& error,int64_t budget){
  const auto initial_bytes=p.retained_tensor_bytes();
  if(!w.count("summary.matrix")||w.at("summary.matrix").dim()!=3)throw std::invalid_argument("missing recurrent summary bank");
  const auto banks=w.at("summary.matrix").size(0),rows=raw.floating.size(0);auto shapes=weighted_recurrent_parameter_shapes(banks,c);
  if(action!=WeightedRecurrentAction::Update&&action!=WeightedRecurrentAction::Memory&&action!=WeightedRecurrentAction::Full)
    throw std::invalid_argument("invalid recurrent action");
  if(w.size()!=shapes.size())throw std::invalid_argument("recurrent parameter roles differ from configuration");
  for(const auto& [name,shape]:shapes){if(!w.count(name))throw std::invalid_argument("missing recurrent parameter: "+name);check(w.at(name),raw.floating.device(),at::kFloat,shape);}
  check(content,raw.floating.device(),at::kFloat,{rows,c.width});
  if(raw.floating.scalar_type()==at::kHalf){
    c.validate();if(!raw.integer.defined()||raw.integer.dim()!=2||raw.integer.size(1)<4)
      throw std::invalid_argument("invalid half recurrent metadata");
    WeightedRawLayout layout(c,raw.integer.size(1)-4);validate_weighted_raw_buffers(layout,raw);
    const long double extra=rows*(16.L*layout.floating_width()+24.L*layout.integer_width()+16.L*std::max<int64_t>(1,layout.capacity())+32)+64.L*layout.fields().size();
    if(budget<1||extra>=budget)throw std::invalid_argument("half recurrent tensor budget exceeded");
    auto values=at::empty(raw.floating.sizes(),raw.floating.options().dtype(at::kFloat));p.cast(raw.floating,values);
    auto t=append_weighted_recurrent(p,c,action,{values,raw.integer},times,mass,content,owners,present,w,error,budget-static_cast<int64_t>(extra));
    if(t.update){
      t.update->result=append_weighted_raw_half_store(p,layout,t.update->result,error,static_cast<int64_t>(extra)-4*values.numel());
      for(const auto& field:layout.fields())t.fields[field.name]=columns(p,t.update->result.floating,field.offset,field.width*(field.ragged?layout.capacity():1));
    }
    if(p.retained_tensor_bytes()-initial_bytes>budget)throw std::invalid_argument("complete half recurrent tensor budget exceeded");
    return t;
  }
  auto snapshot=append_weighted_recurrent_snapshot(p,c,raw,times,mass,owners,present,banks,action==WeightedRecurrentAction::Update,error,budget);
  auto graph=std::make_shared<WeightedRowGraph>(p,snapshot.owners,banks,error,budget);
  WeightedRecurrentTape t{c,action,snapshot,graph};t.parameters=w;auto& g=*graph;const auto& l=snapshot.layout;
  t.inputs["mass"]=g.input(mass.reshape({rows,1}));t.inputs["content"]=g.input(content);
  for(const auto& field:l.fields())if(field.name!="value")t.inputs[field.name]=g.input(columns(p,snapshot.state.floating,field.offset,field.width*(field.ragged?l.capacity():1)),field.name=="W");
  recurrent_algebra(t,w,p,times,error,budget);
  if(action==WeightedRecurrentAction::Update){
    auto proposal=at::zeros_like(raw.floating);p.zero(proposal);for(const auto& [name,index]:t.outputs)insert(p,proposal,l.field(name).offset,g.value(index));
    t.update=append_weighted_raw_update(p,l,raw,times,mass,present,proposal,error,budget);
    for(const auto& field:l.fields())t.fields[field.name]=columns(p,t.update->result.floating,field.offset,field.width*(field.ragged?l.capacity():1));
  }else for(const auto& [name,index]:t.outputs)t.fields[name]=g.value(index);
  if(p.retained_tensor_bytes()-initial_bytes>budget)throw std::invalid_argument("complete recurrent tensor budget exceeded");return t;
}
}
