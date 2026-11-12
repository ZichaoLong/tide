// KV §3: actual snapshot banks, node×B Attention and staged Describe.
#include "weighted_column_groups.h"
#include "weighted_describe_heads.h"
#include "weighted_kv_tensor.h"
#include "tide/packing_work.h"
#include <numeric>
namespace tide::weighted::columns {
using detail::require;using detail::weight;
namespace {
const Layout& bank_layout(const KVReadRequest& r){return r.mode=="content"?r.initial.layout:r.window.bank_layout;}
TensorFields single(const KVReadRequest& q){return read(*q.weights,q.config,q.regional,q.window,q.mode,q.initial);}
std::vector<TensorFields> evaluate(const std::vector<KVReadRequest>& requests,const std::vector<size_t>& ids){
  const auto& c=requests[ids[0]].config;
  std::vector<QueryInputs> qs;std::vector<Tensor> sw,st,lt,shortv,longv;
  std::vector<const NodeWeights*> owners;
  for(auto i:ids){const auto& r=requests[i];const auto& window=r.window;
    std::vector<Index> rows(window.layout.times.size());std::iota(rows.begin(),rows.end(),0);
    qs.push_back(query_inputs(window,bank_layout(r),r.mode=="content"?r.initial.fields:window.bank,rows,r.mode=="old",c.window,c.memory_read=="sum"));
    const auto& q=qs.back();auto shortw=q.weights*detail::passive_decay(q.age.clamp_min(0),c.beta_short,q.u);
    sw.push_back(shortw);st.push_back(shortw.sum(-1));lt.push_back(q.weights.sum(-1));owners.push_back(r.weights);
    shortv.push_back(at::zeros({q.u.size(0),q.u.size(1),c.summary_width},q.u.options()));longv.push_back(shortv.back());
  }
  if(c.memory_read=="sum"){
    std::vector<std::vector<Tensor>> sr,lr;
    for(size_t i=0;i<qs.size();++i){sr.push_back({sw[i],qs[i].raw.at("summaries")});lr.push_back({qs[i].weights,qs[i].raw.at("summaries")});}
    shortv=contract_nodes(Contraction::matmul,sr);longv=contract_nodes(Contraction::matmul,lr);
    for(size_t i=0;i<qs.size();++i){shortv[i]=shortv[i]/st[i].clamp_min(1).unsqueeze(-1);longv[i]=longv[i]/lt[i].clamp_min(1).unsqueeze(-1);}
  }else{
    auto selected=groups::active(qs);
    if(!selected.empty()){
      std::vector<Tensor> us,ks,vs,wq,wk,wv;
      for(auto i:selected){us.push_back(normalize(qs[i].u,c.epsilon));ks.push_back(qs[i].raw.at("keys"));vs.push_back(qs[i].raw.at("values"));
        wq.push_back(weight(*owners[i],"kernel.read_q"));wk.push_back(weight(*owners[i],"kernel.read_k"));wv.push_back(weight(*owners[i],"kernel.read_v"));}
      auto queries=project_nodes(us,wq),keys=project_nodes(ks,wk),values=project_nodes(vs,wv);
      std::vector<std::vector<Tensor>> sr,lr;auto kind=c.read_frequencies.empty()?Contraction::attention:Contraction::logits;
      std::vector<Tensor> logits;
      if(!c.read_frequencies.empty()){std::vector<PositionQuery> positions;
        for(size_t n=0;n<selected.size();++n){auto i=selected[n];const auto& r=requests[ids[i]];
          positions.push_back({queries[n].unsqueeze(1),keys[n].unsqueeze(1),qs[i].age,r.window.layout,
                               bank_layout(r),{c.read_frequencies},c.position_scale,{}});}
        logits=position_logits_many(positions);}
      for(size_t n=0;n<selected.size();++n){auto i=selected[n];auto q=queries[n],k=keys[n],v=values[n];std::vector<Tensor> operands;
        if(!c.read_frequencies.empty()){
          operands={logits[n],v.unsqueeze(1)};
        }else operands={q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1)};
        auto a=operands,b=operands;a.push_back(sw[i]);b.push_back(qs[i].weights);sr.push_back(std::move(a));lr.push_back(std::move(b));
      }
      auto a=contract_nodes(kind,sr),b=contract_nodes(kind,lr);
      for(size_t n=0;n<selected.size();++n){const auto i=selected[n];
        shortv[i]=a[n].select(1,0)*st[i].clamp_max(1).unsqueeze(-1);longv[i]=b[n].select(1,0)*lt[i].clamp_max(1).unsqueeze(-1);}
    }
  }
  std::vector<Tensor> cur,matrices,masses,features;std::vector<detail::Groups> groups;
  for(size_t i=0;i<qs.size();++i){cur.push_back(normalize(qs[i].u,c.epsilon));matrices.push_back(weight(*owners[i],"kernel.summary"));masses.push_back(qs[i].mass);}
  auto current=project_nodes(cur,matrices);
  for(size_t i=0;i<qs.size();++i){auto stat=at::stack({qs[i].mass.log1p(),st[i].log1p(),lt[i].log1p()},-1);
    groups.push_back({cur[i],at::cat({shortv[i],longv[i]},-1),stat});
    features.push_back(at::cat({cur[i],stat.slice(-1,0,1),shortv[i],longv[i],stat.slice(-1,1),current[i]*shortv[i],current[i]*longv[i]},-1));}
  auto out=query::describe_nodes(owners,requests[ids[0]].regional,masses,features,groups,c.epsilon);
  for(size_t i=0;i<out.size();++i)for(auto& [name,t]:out[i])t=t.flatten(0,1).index_select(0,qs[i].reverse);
  return out;
}
}
std::vector<TensorFields> read_many(const std::vector<KVReadRequest>& requests,Index max_workspace_bytes){
  std::vector<size_t> ids,tensors;std::vector<TensorFields> out(requests.size());
  for(size_t i=0;i<requests.size();++i){const auto& r=requests[i];require(r.weights,"KV Read requires a live parameter owner");
    require(supported(r.config),"unsupported KV Read group");
    require(r.mode=="content"||r.mode=="old"||r.mode=="proposal","invalid KV column Read snapshot");
    require(!r.window.layout.times.empty(),"Read columns require actual rows");
    if(r.mode=="content")require(r.initial.layout.samples()==r.window.layout.samples(),"Read0 initial sample domain mismatch");
    (tensor::eligible(r.config,r.window.content)?tensors:ids).push_back(i);}
  tensor::read_groups(requests,tensors,max_workspace_bytes,out);
  auto same=[&](size_t a,size_t b){const auto& x=requests[a];const auto& y=requests[b];
    return same_configuration(x.config,y.config)&&same_configuration(x.regional,y.regional)&&x.mode==y.mode&&
      groups::same_payload(x.window.content,y.window.content)&&groups::shape(x.window.layout)==groups::shape(y.window.layout)
      &&groups::shape(bank_layout(x))==groups::shape(bank_layout(y));};
  auto cost=[&](size_t i){const auto& r=requests[i];return groups::bytes(*r.weights,r.config,r.window.layout,bank_layout(r),r.window.content.element_size());};
  for(const auto& batch:groups::plan(ids,max_workspace_bytes,same,cost)){
    work::packing::bucket("kv_read_nodes",groups::shape(requests[batch[0]].window.layout).second,batch.size());
    if(batch.size()==1){out[batch[0]]=single(requests[batch[0]]);continue;}
    auto values=evaluate(requests,batch);for(size_t k=0;k<batch.size();++k)out[batch[k]]=std::move(values[k]);}
  return out;
}
}
