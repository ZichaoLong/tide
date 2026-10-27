#include "weighted_flow_vjp.h"
namespace tide::device_online {
namespace {
using namespace weighted_reverse;
void content(DeviceProgram& p,const WeightedRowAdjoint& target,const WeightedRowAdjoint& part,int64_t start,const at::Tensor& error){
  auto value=at::zeros_like(target.value);p.zero(value);weighted_copy(p,value.narrow(1,start,part.value.size(1)),part.value);
  add(p,target,{value,part.connected},at::zeros({value.size(0)},error.options().dtype(at::kLong)),error);
}
void recurrent(DeviceProgram& p,const WeightedFlowNode& n,const WeightedRecurrentGradient& g,const WeightedRawAdjoint& raw,
    const WeightedRowAdjoint& input,std::vector<WeightedParameterPartial>& parameters,const at::Tensor& error){
  add(p,n.layout,raw,{g.raw,g.raw_connected},error);content(p,input,g.mass,0,error);content(p,input,g.content,1,error);
  weighted_reverse::parameters(p,parameters,n.banks,"recurrent.",g.parameters);
}
}
WeightedNodeGradient append_weighted_node_vjp(DeviceProgram& p,TaskServices& services,const WeightedFlowNode& n,
    const std::string& read_mode,const WeightedNodeRoots& roots,const at::Tensor& error,const WeightedFlowBudget& remaining){
  using namespace weighted_reverse;
  if(read_mode!="content"&&read_mode!="old"&&read_mode!="proposal")throw std::invalid_argument("invalid weighted VJP Read snapshot");
  if(!n.aggregate||!n.describe||(!n.kv_full&&!n.recurrent_full))throw std::invalid_argument("weighted stage VJP needs complete forward tapes");
  const auto rows=n.input.present.numel();auto owners=at::zeros({rows},error.options().dtype(at::kLong));
  WeightedNodeGradient out;out.old=zeros(p,n.layout,n.old.floating);out.initial=zeros(p,n.layout,n.initial.floating);
  auto proposal=zeros(p,n.layout,n.proposal.floating);auto input=zeros(p,n.aggregate->output);
  add(p,n.layout,proposal,roots.proposal,error);add(p,input,roots.content,owners,error);
  auto full=zeros(p,n.full);add(p,full,roots.full,owners,error);
  auto role=[&](const std::string& name,const at::Tensor& value,const at::Tensor& flag){parameter(p,out.parameters,n.banks,name,value,flag);};
  auto& fworker=services.begin();
  if(n.kv_full){auto g=append_weighted_kv_full_vjp(fworker,*n.kv_full,full.value,full.connected,error,remaining(fworker));services.end(fworker);
    add(p,n.layout,proposal,{g.raw,g.raw_connected},error);content(p,input,{g.input,g.input_connected},1,error);
    role("full.query",g.parameters.query,g.query_connected);role("full.out",g.parameters.residual.out,g.out_connected);
    role("full.gate",g.parameters.residual.gate,g.gate_connected);role("full.up",g.parameters.residual.up,g.up_connected);role("full.down",g.parameters.residual.down,g.down_connected);
  }else{auto g=append_weighted_recurrent_vjp(fworker,*n.recurrent_full,{{"G",full}},error,remaining(fworker));services.end(fworker);
    recurrent(p,n,g,proposal,input,out.parameters,error);}
  auto& dworker=services.begin();auto describe=append_weighted_describe_vjp(dworker,*n.describe,roots.describe,error,remaining(dworker));services.end(dworker);
  parameters(p,out.parameters,n.banks,"describe.",describe.parameters);
  content(p,input,describe.inputs.at("mass"),0,error);content(p,input,describe.inputs.at("content"),1,error);
  const auto read=read_mode=="content"?out.initial:read_mode=="old"?out.old:proposal;
  auto& mworker=services.begin();
  if(n.kv_sum||n.kv_query){const auto ds=std::get<weighted::KVConfig>(n.binding.memory).summary_width;
    auto memory=at::zeros({rows,2,ds},input.value.options()),total=at::zeros({rows,2},input.value.options());
    auto flags=at::zeros({rows,4},error.options().dtype(at::kBool));
    const std::string names[]{"short_memory","long_memory","short_total","long_total"};
    for(int j=0;j<4;++j){const auto& root=describe.inputs.at(names[j]);
      weighted_copy(mworker,j<2?memory.select(1,j):total.narrow(1,j-2,1),root.value);
      weighted_copy(mworker,flags.select(1,j),root.connected);}
    if(n.kv_sum){auto g=append_weighted_kv_sum_vjp(mworker,*n.kv_sum,memory,total,flags,error,remaining(mworker));services.end(mworker);
      add(p,n.layout,read,{g.raw,g.connected},error);
    }else{auto g=append_weighted_kv_query_vjp(mworker,*n.kv_query,memory,total,flags,error,remaining(mworker));services.end(mworker);
      add(p,n.layout,read,{g.raw,g.raw_connected},error);content(p,input,{g.input,g.input_connected},1,error);
      role("mem.query",g.parameters.query,g.query_connected);role("mem.key",g.parameters.key,g.key_connected);role("mem.value",g.parameters.value,g.value_connected);}
  }else{WeightedFieldAdjoints fields;for(const auto& name:{"short_memory","long_memory","short_total","long_total"})fields[name]=describe.inputs.at(name);
    auto g=append_weighted_recurrent_vjp(mworker,*n.recurrent_mem,fields,error,remaining(mworker));services.end(mworker);
    recurrent(p,n,g,read,input,out.parameters,error);}
  auto& uworker=services.begin();
  if(n.kv_update){auto g=append_weighted_kv_update_vjp(uworker,*n.kv_update,proposal.value,proposal.connected,error,remaining(uworker));services.end(uworker);
    add(p,n.layout,out.old,{g.old_values,g.old_connected},error);add(p,input,{g.content,g.content_connected},owners,error);
    role("upd.base",g.parameters.base,g.kv_connected);role("upd.left",g.parameters.left,g.kv_connected);
    role("upd.right",g.parameters.right,g.kv_connected);role("upd.summary",g.parameters.summary,g.summary_connected);
  }else{WeightedFieldAdjoints fields;for(size_t j=0;j<n.layout.fields().size();++j){const auto& f=n.layout.fields()[j];
      const auto width=f.width*(f.ragged?n.layout.capacity():1);
      fields[f.name]={weighted_columns(uworker,proposal.value,f.offset,width),weighted_columns(uworker,proposal.connected,j,1).reshape({rows})};}
    auto g=append_weighted_recurrent_vjp(uworker,*n.recurrent_update,fields,error,remaining(uworker));services.end(uworker);
    recurrent(p,n,g,out.old,input,out.parameters,error);}
  auto& aworker=services.begin();auto g=append_weighted_aggregate_vjp(aworker,*n.aggregate,input.value,input.connected,error,remaining(aworker));services.end(aworker);
  out.payload=g.payload;out.connected=g.connected;source(p,services,n,g,out.parameters,error,remaining);
  remaining(p);return out;
}
}
