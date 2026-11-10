// KV §4: B/H/Q/K matrix contraction without per-query K/V expansion in forward.
#include "weighted_bank_attention.h"
#include "tide/packing_work.h"
#include "tide/autograd_census.h"
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>

namespace tide::weighted::query {
namespace {
using torch::autograd::variable_list;
struct Plan {
  std::vector<std::array<Index,4>> groups; // source, key start, value start, length
  std::vector<Index> lengths,starts,inverse;
  Index width=0,queries=0,length=1,unique=0;
  Tensor rows,reverse;
  Tensor bank(const variable_list& xs,bool value)const{
    variable_list parts;
    for(const auto& g:groups){
      auto part=xs[g[0]].as_strided({g[3],width},{width,1},g[value?2:1]*width);
      parts.push_back(length==g[3]?part:at::cat({part,at::zeros({length-g[3],width},part.options())}));
    }
    return parts.size()==1?parts[0].unsqueeze(0):at::stack(parts);
  }
  Tensor group(const Tensor& t)const{
    auto shape=rows.sizes().vec();shape.insert(shape.end(),t.sizes().begin()+1,t.sizes().end());
    return at::cat({t,at::zeros_like(t.slice(0,0,1))}).index_select(0,rows.flatten()).reshape(shape);
  }
  Tensor ungroup(const Tensor& t)const{return t.flatten(0,1).index_select(0,reverse);}
  Tensor weight_index(Index columns)const{
    auto start=starts,size=lengths;start.push_back(0);size.push_back(0);
    auto a=at::tensor(start,rows.options()).index_select(0,rows.flatten()).reshape_as(rows);
    auto n=at::tensor(size,rows.options()).index_select(0,rows.flatten()).reshape_as(rows);
    auto col=at::arange(length,rows.options()).reshape({1,1,-1})-a.unsqueeze(-1);
    auto present=(col>=0)&(col<n.unsqueeze(-1));
    return at::where(present,rows.unsqueeze(-1)*columns+col,static_cast<int64_t>(lengths.size())*columns);
  }
};
std::optional<Plan> make_plan(const variable_list& keys,const variable_list& values){
  if(keys.size()<2||keys.size()!=values.size()||keys[0].dim()!=2)return {};
  const auto& ref=keys[0];Plan p;p.width=ref.size(1);
  if(!p.width||(ref.scalar_type()!=at::kFloat&&ref.scalar_type()!=at::kDouble))return {};
  using Identity=std::tuple<const c10::StorageImpl*,const c10::StorageImpl*,Index>;
  std::map<Identity,size_t> identities;std::vector<std::vector<std::array<Index,3>>> members;
  Index total=0,max_row=1;
  for(size_t i=0;i<keys.size();++i){
    const auto& k=keys[i];const auto& v=values[i];if(k.sizes()!=v.sizes())return {};
    for(const auto& t:{k,v})if(t.dim()!=2||t.size(1)!=p.width||t.device()!=ref.device()||t.scalar_type()!=ref.scalar_type()
        ||!t.is_contiguous()||t.is_conj()||t.is_neg()||t.storage_offset()%p.width)return {};
    auto a=k.storage_offset()/p.width,b=v.storage_offset()/p.width;
    Identity id{k.storage().unsafeGetStorageImpl(),v.storage().unsafeGetStorageImpl(),b-a};
    if(!identities.count(id)){identities[id]=members.size();members.emplace_back();}
    members[identities.at(id)].push_back({static_cast<Index>(i),a,b});
    p.lengths.push_back(k.size(0));total+=k.size(0);max_row=std::max(max_row,k.size(0));
  }
  for(const auto& m:members)p.queries=std::max(p.queries,static_cast<Index>(m.size()));
  p.starts.resize(keys.size());p.inverse.resize(keys.size());std::vector<Index> rows;
  for(size_t g=0;g<members.size();++g){
    const auto& m=members[g];Index start=m[0][1],end=start;
    for(const auto& r:m){start=std::min(start,r[1]);end=std::max(end,r[1]+p.lengths[r[0]]);}
    const auto length=end-start;p.length=std::max(p.length,length);p.unique+=length;
    p.groups.push_back({m[0][0],start,start+m[0][2]-m[0][1],length});
    for(size_t j=0;j<m.size();++j){p.starts[m[j][0]]=m[j][1]-start;p.inverse[m[j][0]]=g*p.queries+j;}
  }
  if(p.unique>=total||static_cast<long double>(p.groups.size())*p.queries*p.length>2.L*keys.size()*max_row)return {};
  for(const auto& m:members){
    for(const auto& row:m)rows.push_back(row[0]);
    for(size_t j=m.size();j<size_t(p.queries);++j)rows.push_back(keys.size());
  }
  auto opts=ref.options().dtype(at::kLong);
  p.rows=at::tensor(rows,opts).reshape({static_cast<Index>(members.size()),p.queries});p.reverse=at::tensor(p.inverse,opts);
  return p;
}
class AttentionVjp final:public torch::autograd::Node {
 public:
  Plan plan;Tensor q,k,v,w,prob;
  std::vector<torch::autograd::SavedVariable> originals;
  variable_list apply(variable_list&& bars)override{
    if(at::GradMode::is_enabled())throw std::invalid_argument("KV bank attention supports first-order VJP only");
    if(originals.empty())throw std::runtime_error("bank attention graph released; use retain_graph");
    variable_list xs;for(auto& s:originals)xs.push_back(s.unpack());variable_list result(xs.size());
    if(!bars[0].defined())return result;
    const auto n=xs[0].size(0),heads=xs[0].size(1),d=xs[0].size(2);
    const auto& p=plan;auto bar=p.group(bars[0]).transpose(1,2),positive=w>0;
    auto first=positive.to(at::kLong).argmax(-1).unsqueeze(1).unsqueeze(-1).unsqueeze(-1);
    auto expanded=v.unsqueeze(2).expand({-1,-1,p.queries,-1,-1});
    auto anchor=expanded.gather(-2,first.expand({-1,heads,-1,1,d}));
    auto delta=at::where(positive.unsqueeze(1).unsqueeze(-1),expanded,anchor)-anchor;
    auto centered=delta-(prob.unsqueeze(-1)*delta).sum(-2,true);
    auto dl=prob*(centered*bar.unsqueeze(-2)).sum(-1);
    if(should_compute_output(0))result[0]=p.ungroup((at::matmul(dl,k)/std::sqrt(double(d))).transpose(1,2));
    if(should_compute_output(1)){
      auto dw=p.ungroup(dl.sum(1)/at::where(positive,w,at::ones_like(w)));variable_list rows;
      for(Index i=0;i<n;++i)rows.push_back(at::cat({dw[i].slice(0,p.starts[i],p.starts[i]+p.lengths[i]),
        at::zeros({xs[1].size(1)-p.lengths[i]},xs[1].options())}));
      result[1]=at::stack(rows);
    }
    bool key=false,value=false;
    for(Index i=0;i<n;++i){key|=should_compute_output(2+i);value|=should_compute_output(2+n+i);}
    auto rows=[&](const Tensor& t){return p.ungroup(t.transpose(1,2)).transpose(1,2).reshape({n,p.length,heads*d});};
    auto dk=key?rows(dl.unsqueeze(-1)*q.unsqueeze(-2)/std::sqrt(double(d))):Tensor{};
    auto dv=value?rows(prob.unsqueeze(-1)*bar.unsqueeze(-2)):Tensor{};
    for(Index i=0;i<n;++i){
      if(should_compute_output(2+i))result[2+i]=dk[i].slice(0,p.starts[i],p.starts[i]+p.lengths[i]);
      if(should_compute_output(2+n+i))result[2+n+i]=dv[i].slice(0,p.starts[i],p.starts[i]+p.lengths[i]);
    }
    return result;
  }
  void release_variables()override{originals.clear();q=k=v=w=prob=Tensor{};plan.rows=plan.reverse=Tensor{};}
  std::string name()const override{return "tide::weighted::BankAttentionVjp";}
};
}
Tensor bank_attention(const Tensor& queries,const variable_list& keys,const variable_list& values,const Tensor& weights){
  auto planned=make_plan(keys,values);if(!planned)return {};
  if(!(weights>0).any(-1).all().item<bool>())return {}; // preserve empty structural branches
  auto node=std::make_shared<AttentionVjp>();node->plan=std::move(*planned);const auto& p=node->plan;
  const Index heads=queries.size(1),d=queries.size(2);Tensor output;
  {
    at::NoGradGuard guard;node->q=p.group(queries).permute({0,2,1,3});
    node->k=p.bank(keys,false).reshape({-1,p.length,heads,d}).transpose(1,2);
    node->v=p.bank(values,true).reshape({-1,p.length,heads,d}).transpose(1,2);
    auto index=p.weight_index(weights.size(1));
    node->w=at::cat({weights.flatten(),at::zeros({1},weights.options())}).index_select(0,index.flatten()).reshape(index.sizes());
    auto positive=node->w>0,used=positive.any(1).unsqueeze(1).unsqueeze(-1);
    node->k=at::where(used,node->k,at::zeros_like(node->k));node->v=at::where(used,node->v,at::zeros_like(node->v));
    auto safe=at::where(positive,node->w,at::ones_like(node->w));
    auto logits=at::matmul(node->q,node->k.transpose(-1,-2))/std::sqrt(double(d))+safe.log().unsqueeze(1);
    logits=logits.masked_fill(~positive.unsqueeze(1),-std::numeric_limits<double>::infinity());
    auto active=positive.any(-1).unsqueeze(1).unsqueeze(-1);
    node->prob=at::where(active,logits,at::zeros_like(logits)).softmax(-1);
    node->prob=at::where(active,node->prob,at::zeros_like(node->prob));
    output=p.ungroup(at::matmul(node->prob,node->v).transpose(1,2));
  }
  work::packing::projection("kv_bank_attention",keys.size());
  variable_list inputs{queries,weights};inputs.insert(inputs.end(),keys.begin(),keys.end());inputs.insert(inputs.end(),values.begin(),values.end());
  bool grad=false;for(const auto& t:inputs)grad|=t.requires_grad();
  if(at::GradMode::is_enabled()&&grad){
    node->set_next_edges(torch::autograd::collect_next_edges(inputs));for(const auto& t:inputs)node->originals.emplace_back(t,false);
    if(autograd_census::enabled()){autograd_census::saved(inputs);autograd_census::saved({node->q,node->k,node->v,node->w,node->prob});}
    torch::autograd::set_history(output,node);
  }
  return output;
}
}
