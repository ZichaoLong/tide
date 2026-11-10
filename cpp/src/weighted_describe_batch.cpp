#include "weighted_private_rows.h"
#include "tide/operator_profile.h"
#include <ATen/record_function.h>
// Architecture §3: batched Mem and capability-selected shared projections.
#include "weighted_query_batch.h"
#include "weighted_scoring.h"
#include <type_traits>

namespace tide::weighted::query {
using namespace detail;
namespace {
template<class Config> std::vector<TensorFields> evaluate(const NodeWeights& w,const std::vector<ReadInput>& requests,const Config& c,const RegionalConfig& r,int dense,bool batch_describe){
  if(requests.empty())return {};
  RECORD_FUNCTION("tide::stage::describe", std::vector<c10::IValue>());
  op_profile::Scope stage_profile(op_profile::QueryMath);
  States states;Times times;std::vector<Tensor> received;
  for(const auto& request:requests){
    states.push_back(request.state?*request.state:w.kernel->initial(w));times.push_back(request.time);received.push_back(request.content.value);
  }
  auto contents=private_rows::join(received,true),masses=contents.select(1,0),input=contents.slice(1,1);
  std::array<Tensor,4> memories;
  if constexpr(std::is_same_v<Config,KVConfig>)memories=memory(w,states,times,input,c,dense);
  else memories=memory(w,states,times,input,c);
  if((batch_describe||dense) && r.local_score=="cat" && r.region_score=="ind" && r.selection_mode=="regional_total"
      && r.region_control_mode=="context" && r.region_evidence=="cat"){
    using packed::project;
    auto cur=normalize(input,c.epsilon),p=project(cur,weight(w,"kernel.summary"));
    auto stat=at::stack({masses.log1p(),memories[2].log1p(),memories[3].log1p()},-1);
    auto f=at::cat({cur,stat.slice(1,0,1),memories[0],memories[1],stat.slice(1,1),p*memories[0],p*memories[1]},-1);
    auto score=weight(w,"read_program.score.bias")+(at::silu(project(f,weight(w,"read_program.score.matrix"))+
      weight(w,"read_program.score.hidden_bias"))*weight(w,"read_program.score.vector")).sum(-1);
    auto evidence=normalize(project(f,weight(w,"read_program.evidence_weight"))+weight(w,"read_program.evidence_bias"),c.epsilon);
    auto recv=masses.clamp_max(1),mu=(r.gamma_max*recv).clamp_max(1);
    // One split transpose per field; preserve distinct public field/row roots.
    auto recv_rows=recv.unbind(),mu_rows=mu.unbind(),score_rows=score.unbind(),evidence_rows=evidence.unbind();
    std::vector<TensorFields> result;
    for(size_t i=0;i<requests.size();++i)result.push_back({{"w",recv_rows[i]},{"mu",mu_rows[i]},{"score",score_rows[i]},{"evidence",evidence_rows[i]}});
    return result;
  }
  std::vector<TensorFields> result;
  for(size_t i=0;i<requests.size();++i){
    auto mass=masses[i],u=input[i];
    std::array<Tensor,4> mem{memories[0][i],memories[1][i],memories[2][i],memories[3][i]};
    auto p=current(w,u,c);
    Groups groups{normalize(u,c.epsilon),at::cat({mem[0],mem[1]}),at::stack({mass.log1p(),mem[2].log1p(),mem[3].log1p()})};
    auto features=at::cat({groups[0],groups[2].slice(0,0,1),mem[0],mem[1],groups[2].slice(0,1),p*mem[0],p*mem[1]});
    auto recv=mass.clamp_max(1);Tensor mu;
    if(r.regional())mu=(r.gamma_max*recv).clamp_max(1);
    else {
      auto gamma=at::full({},r.fixed_gain.value_or(1),mass.options());
      if(r.send_gain=="context")gamma=1+(r.gamma_max-1)*head(w.extra,"read_program.gain",features).sigmoid();
      else if(r.gain_logit)gamma=1+(r.gamma_max-1)*weight(w,"read_program.gain_logit").sigmoid();
      mu=(gamma*mass).clamp_max(1);
    }
    TensorFields fields{{"w",recv},{"mu",mu},{"score",local_score(w,r,features,groups,c.epsilon)}};
    if(r.regional())fields["evidence"]=region_evidence(w,r,features,groups,c.epsilon);
    if(r.region_score=="att"){
      fields["x"]=at::matmul(weight(w,"read_program.current_weight"),groups[0])+weight(w,"read_program.current_bias");
      fields["key"]=memory_key(w,r,groups,c.epsilon);
    }
    result.push_back(std::move(fields));
  }
  return result;
}
}
std::vector<TensorFields> describe(const NodeWeights& w,const std::vector<ReadInput>& requests,const KVConfig& c,const RegionalConfig& r,int dense,bool batch_describe){return evaluate(w,requests,c,r,dense,batch_describe);}
std::vector<TensorFields> describe(const NodeWeights& w,const std::vector<ReadInput>& requests,const RecurrentConfig& c,const RegionalConfig& r,int dense,bool batch_describe){return evaluate(w,requests,c,r,dense,batch_describe);}
}
