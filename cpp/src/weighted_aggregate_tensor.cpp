// Physical-atom mean and centered VJP on G×row×source tensors.
#include "weighted_aggregate_tensor.h"
#include "tide/autograd_census.h"
#include "tide/packing_work.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>

namespace tide::weighted::columns {
namespace {
using torch::autograd::variable_list;
struct Layout {
  Index groups=0,rows=0,sources=0,width=0;
  std::vector<Index> lengths,owners,row_counts;
  Tensor index,inverse,singleton;
};
Layout layout(const std::vector<const AggregateTensorRequest*>& requests){
  Layout p;p.groups=requests.size();p.width=requests[0]->values[0].size(1);Index total=0;
  for(const auto* r:requests){p.rows=std::max(p.rows,static_cast<Index>(r->rows.size()));p.sources=std::max(p.sources,static_cast<Index>(r->sources.size()));
    for(const auto* source:r->sources)total+=source->rows.size();}
  std::vector<Index> index(p.groups*p.rows*p.sources,total),inverse,singleton;Index offset=0;
  for(Index g=0;g<p.groups;++g){const auto& r=*requests[g];p.row_counts.push_back(r.rows.size());singleton.push_back(r.sources.size()==1);
    for(size_t a=0;a<r.sources.size();++a){const auto& rows=r.sources[a]->rows;p.lengths.push_back(rows.size());p.owners.push_back(g);
      for(size_t j=0;j<rows.size();++j){const auto row=std::lower_bound(r.rows.begin(),r.rows.end(),rows[j])-r.rows.begin();
        const auto slot=(g*p.rows+row)*p.sources+a;index[slot]=offset+j;inverse.push_back(slot);}
      offset+=rows.size();}
  }
  const auto device=requests[0]->values[0].device();
  p.index=indices(index,device).reshape({p.groups,p.rows,p.sources});p.inverse=indices(inverse,device);
  p.singleton=indices(singleton,device).to(at::kBool).reshape({p.groups,1,1});return p;
}
class AggregateVjp final:public torch::autograd::Node {
 public:
  Layout layout;
  std::vector<torch::autograd::SavedVariable> originals,numeric;
  std::vector<std::pair<Index,bool>> outputs; // owner, true for content.
  variable_list apply(variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("aggregate tensor supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("aggregate tensor graph released");
    variable_list inputs,values;for(auto& t:originals)inputs.push_back(t.unpack());for(auto& t:numeric)values.push_back(t.unpack());
    const auto& p=layout;auto mass=values[0],value=values[1],total=values[2],mean=values[3],positive=values[4];
    auto gm=at::zeros(total.sizes(),total.options()),gu=at::zeros(mean.sizes(),mean.options());
    std::vector<bool> used_mass(p.groups,false),used_value(p.groups,false);std::vector<Index> means(p.groups,0);
    for(size_t j=0;j<bars.size();++j)if(bars[j].defined()){const auto [g,is_value]=outputs[j];
      if(is_value){gu[g].narrow(0,0,p.row_counts[g]).copy_(bars[j]);used_value[g]=true;means[g]=1;}
      else {gm[g].narrow(0,0,p.row_counts[g]).copy_(bars[j]);used_mass[g]=true;}}
    if(std::none_of(used_value.begin(),used_value.end(),[](bool x){return x;})){
      auto selected=gm.unsqueeze(-1).expand({p.groups,p.rows,p.sources}).reshape({-1}).index_select(0,p.inverse).split_with_sizes(p.lengths);
      variable_list result(inputs.size());for(size_t j=0;j<p.owners.size();++j)
        if(used_mass[p.owners[j]]&&should_compute_output(2*j))result[2*j]=selected[j]*(inputs[2*j]>=0);
      return result;
    }
    auto live=positive&indices(means,total.device()).to(at::kBool).unsqueeze(-1);
    gu=at::where(live.unsqueeze(-1),gu,at::zeros({},gu.options()));
    auto denominator=at::where(positive,total,at::ones_like(total));
    auto delta=at::where(live.unsqueeze(-1).unsqueeze(-1),value-mean.unsqueeze(-2),at::zeros({},value.options()));
    auto dm=(delta*gu.unsqueeze(-2)).sum(-1)/denominator.unsqueeze(-1);
    dm=at::where(p.singleton,at::zeros({},dm.options()),dm)+gm.unsqueeze(-1);
    auto dv=(mass/denominator.unsqueeze(-1)).unsqueeze(-1)*gu.unsqueeze(-2);
    // Every physical atom-row has one slot. Repeated tensor aliases still have
    // distinct slots/edges; inverse gather avoids any atomic scatter reduction.
    auto masses=dm.flatten().index_select(0,p.inverse).split_with_sizes(p.lengths);
    auto vectors=dv.flatten(0,2).index_select(0,p.inverse).split_with_sizes(p.lengths);
    variable_list result(inputs.size());
    for(size_t j=0;j<p.owners.size();++j){const auto g=p.owners[j];
      if((used_mass[g]||used_value[g])&&should_compute_output(2*j))result[2*j]=masses[j]*(inputs[2*j]>=0);
      if(used_value[g]&&should_compute_output(2*j+1))result[2*j+1]=vectors[j];}
    return result;
  }
  void release_variables()override{originals.clear();numeric.clear();layout=Layout{};outputs.clear();}
  std::string name()const override{return "tide::weighted::AggregateTensorVjp";}
};
std::vector<Received> evaluate(const std::vector<const AggregateTensorRequest*>& requests){
  variable_list inputs,masses,vectors;bool grad=false;
  for(const auto* r:requests)for(size_t i=0;i<r->sources.size();++i){
    inputs.insert(inputs.end(),{r->sources[i]->strength,r->values[i]});masses.push_back(inputs[inputs.size()-2]);vectors.push_back(inputs.back());}
  if(at::GradMode::is_enabled())for(const auto& t:inputs)grad|=t.requires_grad();
  at::NoGradGuard guard;auto p=layout(requests);
  masses.push_back(at::zeros({1},masses[0].options()));vectors.push_back(at::zeros({1,p.width},vectors[0].options()));
  auto mass=at::cat(masses).index_select(0,p.index.flatten()).reshape({p.groups,p.rows,p.sources}).clamp_min(0);
  auto value=at::cat(vectors).index_select(0,p.index.flatten()).reshape({p.groups,p.rows,p.sources,p.width});
  auto total=mass.sum(-1),positive=total>0,denominator=at::where(positive,total,at::ones_like(total));
  value=at::where(positive.unsqueeze(-1).unsqueeze(-1),value,at::zeros({},value.options()));
  auto mean=(mass.unsqueeze(-1)*value).sum(-2)/denominator.unsqueeze(-1);
  std::shared_ptr<AggregateVjp> node;Tensor live;
  if(grad){node=std::make_shared<AggregateVjp>();node->layout=p;node->set_next_edges(torch::autograd::collect_next_edges(inputs));
    for(const auto& t:inputs)node->originals.emplace_back(t,false);
    for(const auto& t:{mass,value,total,mean,positive})node->numeric.emplace_back(t,false);
    live=positive.any(-1).to(at::kCPU).contiguous();
    if(autograd_census::enabled()){autograd_census::saved(inputs);autograd_census::saved({mass,value,total,mean,positive,p.index,p.inverse,p.singleton});}}
  std::vector<Received> out;
  for(Index g=0;g<p.groups;++g){auto m=total[g].narrow(0,0,p.row_counts[g]),u=mean[g].narrow(0,0,p.row_counts[g]);bool dm=false,du=false;
    if(grad){for(size_t j=0;j<p.owners.size();++j)if(p.owners[j]==g){dm|=inputs[2*j].requires_grad();du|=inputs[2*j].requires_grad()||inputs[2*j+1].requires_grad();}
      du=du&&live.const_data_ptr<bool>()[g];
      if(dm){node->outputs.emplace_back(g,false);torch::autograd::set_history(m,node);}
      if(du){node->outputs.emplace_back(g,true);torch::autograd::set_history(u,node);}}
    out.push_back({requests[g]->rows,m,u});}
  work::packing::bucket("aggregate_tensor",p.rows*p.sources,p.groups);return out;
}
void validate(const AggregateTensorRequest& r){
  detail::require(r.sources.size()==r.values.size()&&!r.sources.empty(),"invalid aggregate tensor atom columns");
  Index last=-1;for(auto row:r.rows){detail::require(row>last,"aggregate tensor rows must be sorted unique nonnegative");last=row;}
  const auto& like=r.values[0];detail::require(like.dim()==2&&like.size(1)>0,"invalid aggregate tensor value shape");
  for(size_t i=0;i<r.sources.size();++i){const auto& source=*r.sources[i];const auto& z=r.values[i];const Index n=source.rows.size();
    detail::require(n>0&&source.strength.sizes()==at::IntArrayRef({n})&&z.sizes()==at::IntArrayRef({n,like.size(1)})
      &&z.device()==like.device()&&source.strength.device()==like.device()&&z.scalar_type()==like.scalar_type()
      &&source.strength.scalar_type()==like.scalar_type(),"aggregate tensor source metadata differs");
    Index previous=-1;for(auto row:source.rows){detail::require(row>previous&&std::binary_search(r.rows.begin(),r.rows.end(),row),"aggregate tensor row map invalid");previous=row;}
  }
}
long double bytes(Index count,Index rows,Index sources,const Tensor& like){
  const long double size=like.element_size(),d=like.size(1),g=count,r=rows,s=sources;
  // Source MLP buffers are already owned. Charge mean packing, reverse values,
  // cotangents and forward/inverse indices at the new whole-group padding.
  return g*r*s*((18*d+18)*size+24)+g*r*(6*d+8)*size;
}
}
std::vector<Received> aggregate_tensor_many(const std::vector<AggregateTensorRequest>& requests,Index budget,
    const std::function<Received(const AggregateTensorRequest&)>& fallback){
  detail::require(budget>0,"aggregate tensor workspace must be positive");std::vector<Received> result(requests.size());
  using Key=std::tuple<int,int,int,Index>;std::map<Key,std::vector<size_t>> groups;
  for(size_t i=0;i<requests.size();++i){const auto& r=requests[i];if(r.rows.empty())continue;validate(r);const auto& like=r.values[0];
    if(like.scalar_type()!=at::kFloat&&like.scalar_type()!=at::kDouble){result[i]=fallback(r);continue;}
    groups[{int(like.device().type()),like.device().index(),int(like.scalar_type()),like.size(1)}].push_back(i);}
  for(const auto& [key,ids]:groups){std::vector<size_t> pending;Index rows=0,sources=0;
    auto flush=[&](){if(pending.empty())return;
      if(pending.size()==1)result[pending[0]]=fallback(requests[pending[0]]);
      else {std::vector<const AggregateTensorRequest*> group;for(auto i:pending)group.push_back(&requests[i]);auto values=evaluate(group);
        for(size_t j=0;j<pending.size();++j)result[pending[j]]=std::move(values[j]);}
      pending.clear();rows=sources=0;
    };
    for(auto i:ids){const auto& r=requests[i];const auto count=static_cast<Index>(r.rows.size()),atoms=static_cast<Index>(r.sources.size());
      if(!pending.empty()&&bytes(pending.size()+1,std::max(rows,count),std::max(sources,atoms),r.values[0])>budget)flush();
      if(bytes(1,count,atoms,r.values[0])>budget){result[i]=fallback(r);continue;}
      pending.push_back(i);rows=std::max(rows,count);sources=std::max(sources,atoms);}
    flush();}
  return result;
}
}
