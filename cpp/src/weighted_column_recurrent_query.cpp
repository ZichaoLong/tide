// Recurrent Mem/Full on actual snapshot columns; node and head axes only.
#include "weighted_column_recurrent.h"
#include "weighted_describe_heads.h"
#include <numeric>
namespace tide::weighted::columns {
using detail::require;using detail::weight;
TensorFields read(const NodeWeights& w,const RecurrentConfig& c,const RegionalConfig& r,const RecurrentWindow& window,
                  const std::string& mode,const RecurrentState& initial){
  require(mode=="content"||mode=="old"||mode=="proposal","invalid recurrent Read snapshot");
  require(!window.layout.times.empty(),"Read columns require actual rows");
  std::vector<Index> rows(window.layout.times.size());std::iota(rows.begin(),rows.end(),0);
  auto state=rec::snapshot(window,rows,mode,initial);auto s=rec::effective(state,c);const auto& u=window.content;
  Tensor shortv,longv;
  if(c.memory_read=="state"&&c.basic()){
    auto q=project(normalize(u,c.epsilon),weight(w,"kernel.read_query"));
    auto values=at::matmul(weight(w,"kernel.read_out"),at::matmul(s[0],q.unsqueeze(1).unsqueeze(-1))).squeeze(-1);
    shortv=values.select(1,0);longv=values.select(1,1);
  }else{
    auto values=s[1]/s[2].clamp_min(1).unsqueeze(-1);shortv=values.select(1,0);longv=values.select(1,1);
    if(c.memory_read=="state"){
      auto f=rec::features(w,c,state.layout,u,state.cache_layout,state.cache_values);
      longv=project(at::matmul(s[0],f.query.unsqueeze(-1)).squeeze(-1).flatten(1),weight(w,"kernel.read_out"));
    }
  }
  auto cur=normalize(u,c.epsilon),current=project(cur,weight(w,"kernel.summary"));
  auto stat=at::stack({window.mass.log1p(),s[2].select(1,0).log1p(),s[2].select(1,1).log1p()},-1);
  detail::Groups groups{cur,at::cat({shortv,longv},-1),stat};
  auto features=at::cat({cur,stat.slice(-1,0,1),shortv,longv,stat.slice(-1,1),current*shortv,current*longv},-1);
  return query::describe_columns(w,r,window.mass,features,groups,c.epsilon);
}
Message full(const NodeWeights& w,const RecurrentConfig& c,const RecurrentWindow& window,const Tensor& controls){
  require(controls.sizes()==window.mass.sizes()&&controls.device()==window.mass.device()
    &&controls.scalar_type()==window.mass.scalar_type()&&at::isfinite(controls).all().item<bool>()
    &&!(controls<0).any().item<bool>(),"invalid recurrent Full controls");
  auto ids=(controls.detach()>0).nonzero().flatten();auto rows=host_ids(ids);
  Message out{rows,controls.index_select(0,ids),at::empty({0,c.width},window.content.options())};if(rows.empty())return out;
  auto u=window.content.index_select(0,ids);auto state=rec::snapshot(window,rows,"proposal",{});
  auto s=rec::effective(state,c);Tensor mix;const std::string p="full_program.content.";
  if(c.basic()){
    auto q=project(normalize(u,c.epsilon),weight(w,p+"query"));
    mix=at::matmul(weight(w,p+"out"),at::matmul(s[0],q.unsqueeze(1).unsqueeze(-1))).squeeze(-1).sum(1);
  }else{
    auto f=rec::features(w,c,state.layout,u,state.cache_layout,state.cache_values);
    auto read=at::matmul(s[0],f.query.unsqueeze(-1)).squeeze(-1);Tensor gated;
    if(c.memory_family=="gated_delta")
      gated=detail::scaled_rms(read,weight(w,p+"rho"),c.output_epsilon).flatten(1)*at::silu(project(normalize(u,c.epsilon),weight(w,p+"output_gate")));
    else{
      auto y=(read+weight(w,p+"direct").unsqueeze(-1)*f.value).flatten(1)*at::silu(f.gate);
      gated=detail::scaled_rms(y,weight(w,p+"rho"),c.output_epsilon);
    }mix=project(gated,weight(w,p+"out"));
  }
  auto x=u+c.alpha_mem*mix,n=normalize(x,c.epsilon);
  out.content=x+c.alpha_ff*project(at::silu(project(n,weight(w,p+"gate")))*project(n,weight(w,p+"up")),weight(w,p+"down"));
  return out;
}
}
