// Architecture §3: batched Mem, scalar operation order for finite descriptor heads.
// Small projection rounding is visible in normalized-head near-zero VJPs.
#include "weighted_query_batch.h"
#include "weighted_scoring.h"

namespace tide::weighted::query {
using namespace detail;
namespace {
template<class Config> std::vector<TensorFields> evaluate(const NodeWeights& w,const std::vector<ReadInput>& requests,const Config& c,const RegionalConfig& r){
  if(requests.empty())return {};
  States states;Times times;std::vector<Tensor> received;
  for(const auto& request:requests){
    states.push_back(request.state?*request.state:w.kernel->initial(w));times.push_back(request.time);received.push_back(request.content.value);
  }
  auto contents=at::stack(received),masses=contents.select(1,0),input=contents.slice(1,1);
  auto memories=memory(w,states,times,input,c);
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
std::vector<TensorFields> describe(const NodeWeights& w,const std::vector<ReadInput>& requests,const KVConfig& c,const RegionalConfig& r){return evaluate(w,requests,c,r);}
std::vector<TensorFields> describe(const NodeWeights& w,const std::vector<ReadInput>& requests,const RecurrentConfig& c,const RegionalConfig& r){return evaluate(w,requests,c,r);}
}
