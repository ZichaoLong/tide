// Recurrent §4.1–4.3: H → H A + B and scalar summary prefix composition.
#include "weighted_packed.h"

namespace tide::weighted::packed {
using detail::weight;
PackedStates recurrent(const NodeWeights& w,const std::vector<State>& old,const PackedSequence& batch,const RecurrentConfig& c){
  Writes plan(old,batch);if(plan.ids.empty())return plan.decode(old,batch,{},0);
  auto x=normalize(plan.input,c.epsilon);
  auto delta=plan.gaps(old);
  auto decay=detail::passive_decay(delta.unsqueeze(-1),{c.lambda_short,0.},x);
  Tensor a,b;bool matrix=false;std::vector<std::map<std::string,Tensor>> caches;
  if(c.basic()){
  auto key=project(x,weight(w,"kernel.key"));key=key/key.norm(2,-1,true).clamp_min(1);
  auto value=project(x,weight(w,"kernel.value"));
  auto retain=at::exp(-plan.receive*at::softplus(at::matmul(x,weight(w,"kernel.retain"))+weight(w,"kernel.retain_bias")));
  auto gain=plan.receive;
  matrix=c.memory_family=="delta";
  if(matrix){
    gain=gain*at::sigmoid(at::matmul(x,weight(w,"kernel.write"))+weight(w,"kernel.write_bias"));
    a=(at::eye(c.key_width,x.options())-gain.reshape({-1,1,1})*key.unsqueeze(-1)*key.unsqueeze(-2)).unsqueeze(1);
    a=a*(decay*retain.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1);
  }else a=(decay*retain.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1);
  b=(gain.reshape({-1,1,1})*value.unsqueeze(-1)*key.unsqueeze(-2)).unsqueeze(1).expand({-1,2,-1,-1});
  }else{
    auto coefficients=gated_coefficients(w,old,plan,x,delta,c);
    a=coefficients.a;b=coefficients.b;matrix=coefficients.matrix;caches=std::move(coefficients.caches);
  }
  auto z=plan.receive.unsqueeze(-1)*project(normalize(plan.input,c.epsilon),weight(w,"kernel.summary"));
  std::vector<std::map<std::string,Tensor>> slots(plan.ids.size());
  for(const auto& [length,owners]:plan.groups){
    auto ids=plan.group_index(length,owners);std::vector<Tensor> ih,iz,iw;
    for(auto i:owners){ih.push_back(old[i].slots.at("H"));iz.push_back(old[i].slots.at("Z"));iw.push_back(old[i].slots.at("W"));}
    auto aa=a.index({ids}),bb=b.index({ids});
    auto H=matrix?matrix_scan(aa,bb,at::stack(ih)):affine_scan(aa,bb,at::stack(ih));
    auto D=decay.index({ids});
    auto Z=affine_scan(D.unsqueeze(-1),z.index({ids}).unsqueeze(-2).expand({-1,-1,2,-1}),at::stack(iz));
    auto W=affine_scan(D,plan.receive.index({ids}).unsqueeze(-1).expand({-1,-1,2}),at::stack(iw));
    for(size_t row=0;row<owners.size();++row)for(Index j=0;j<length;++j){auto k=plan.offsets[owners[row]]+j;
      slots[k]={{"time",plan.times[k]},{"H",H[j][row]},{"Z",Z[j][row]},{"W",W[j][row]}};
      if(!c.basic())slots[k].insert(caches[k].begin(),caches[k].end());
    }
  }
  return plan.decode(old,batch,slots,plan.groups.size());
}
}
