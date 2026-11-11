#include "weighted_kv_block.h"
#include "tide/operator_profile.h"
#include <ATen/record_function.h>
#include <algorithm>
#include <ATen/core/grad_mode.h>
#include <set>

namespace tide::weighted::block {
std::optional<Plan> plan(const std::vector<Row>& rows,const KVConfig& c,bool full){
  RECORD_FUNCTION("tide::stage::query_block_layout", std::vector<c10::IValue>());
  op_profile::Scope profile(op_profile::QueryBind);
  if(rows.size()<2||!c.read_frequencies.empty())return {};
  for(const auto& f:c.full_frequencies)if(!f.empty())return {};
  for(auto s:c.distance_slopes)if(s!=0)return {};
  Plan p;p.rows=rows;p.full=full;
  std::map<const Bank*,size_t> owners;std::vector<std::vector<Index>> groups;
  for(size_t i=0;i<rows.size();++i){
    auto v=view(&rows[i].state);
    if(!v||(c.window&&rows[i].time-v->bank->times[0]>=c.window))return {};
    auto id=v->bank.get();
    if(!owners.count(id)){owners[id]=p.banks.size();p.banks.push_back(v->bank);groups.emplace_back();}
    p.owner.push_back(owners.at(id));p.views.push_back(v);groups[owners.at(id)].push_back(i);
    p.length=std::max(p.length,static_cast<Index>(v->bank->times.size()));
  }
  Index queries=0;for(const auto& g:groups)queries=std::max(queries,static_cast<Index>(g.size()));
  const auto& ref=rows[0].payload;
  if(static_cast<long double>(p.banks.size())*queries*p.length*(4*c.heads+4)*ref.element_size()>32*1024*1024)return {};
  if(at::GradMode::is_enabled()){
    std::map<std::string,std::set<bool>> flags;
    for(const auto& b:p.banks){
      for(const auto& name:{"keys","values","summaries","weights"})flags[std::string("old.")+name].insert(b->old.slots.at(name).requires_grad());
      for(const auto& t:b->values)flags["values"].insert(t.requires_grad());
      for(const auto& t:b->summaries)flags["summaries"].insert(t.requires_grad());
      for(const auto& t:b->contents)flags["contents"].insert(t.requires_grad());
    }
    for(const auto& r:rows){flags["payload"].insert(r.payload.requires_grad());if(full)flags["control"].insert(r.control.requires_grad());}
    for(const auto& [name,f]:flags)if(f.size()>1)return {};
  }
  std::vector<Index> ids,ends,times,stamps,inverse(rows.size());
  for(size_t b=0;b<groups.size();++b){
    const auto& g=groups[b];
    for(Index j=0;j<queries;++j){const bool valid=j<static_cast<Index>(g.size());
      ids.push_back(valid?g[j]:rows.size());ends.push_back(valid?p.views[g[j]]->end:0);times.push_back(valid?rows[g[j]].time:0);
      if(valid)inverse[g[j]]=b*queries+j;}
    const auto& t=p.banks[b]->times;stamps.insert(stamps.end(),t.begin(),t.end());stamps.insert(stamps.end(),p.length-t.size(),0);
  }
  at::NoGradGuard no_grad;auto opts=ref.options().dtype(at::kLong);const Index b=p.banks.size();
  p.indices=at::tensor(ids,opts).reshape({b,queries});p.reverse=at::tensor(inverse,opts);
  auto end=at::tensor(ends,opts).reshape({b,queries}),time=at::tensor(times,opts).reshape({b,queries});
  auto stamp=at::tensor(stamps,opts).reshape({b,p.length});
  p.ages=(time.unsqueeze(-1)-stamp.unsqueeze(1)).clamp_min(0);
  p.visible=(at::arange(p.length,opts).reshape({1,1,-1})<end.unsqueeze(-1))&(time.unsqueeze(-1)>=stamp.unsqueeze(1));
  if(!full){
    std::vector<Tensor> banks;
    for(const auto& bank:p.banks){auto raw=at::cat({bank->old.slots.at("weights"),at::stack(bank->contents).select(1,0).clamp_max(1)});
      banks.push_back(at::cat({raw,at::zeros({p.length-raw.numel()},raw.options())}));}
    auto raw=at::stack(banks);auto w=at::where(p.visible,raw.unsqueeze(1),at::zeros({},ref.options()));
    auto shortw=w*detail::passive_decay(p.ages,c.beta_short,w);
    if(((w>0)&(shortw==0)).any().item<bool>())return {};
  }
  return p;
}
}
