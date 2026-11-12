// KV §1.2: raw ragged banks and terminal retention, independently of queries.
#include "weighted_column_layout.h"
#include "weighted_row_linear.h"
#include "weighted_kv_tensor.h"
#include <numeric>
namespace tide::weighted::columns {
using detail::require;using detail::weight;
void Layout::validate()const {
  require(offsets.size()>=2&&offsets.front()==0&&offsets.back()==static_cast<Index>(times.size()),"invalid column offsets");
  for(Index b=0;b<samples();++b)
    require(offsets[b]>=0&&offsets[b]<=offsets[b+1]&&offsets[b+1]<=static_cast<Index>(times.size()),"invalid column offset bounds");
  for(Index b=0;b<samples();++b){
    require(offsets[b]<=offsets[b+1],"decreasing column offsets");
    for(Index j=offsets[b];j<offsets[b+1];++j)
      require(times[j]>=0&&(j==offsets[b]||times[j]>times[j-1]),"column times must strictly increase");
  }
}
Layout Layout::select(const std::vector<Index>& rows)const {
  Layout out;out.offsets={0};Index b=0,previous=-1;
  for(auto row:rows){
    require(row>previous&&row>=0&&row<static_cast<Index>(times.size()),"column rows must be sorted, unique and in range");
    while(b<samples()&&row>=offsets[b+1]){out.offsets.push_back(out.times.size());++b;}
    out.times.push_back(times[row]);previous=row;
  }
  while(b++<samples())out.offsets.push_back(out.times.size());
  return out;
}
KVState KVState::from_states(const std::vector<State>& states){
  require(!states.empty(),"KV columns require samples");KVState out;out.layout.offsets={0};
  std::map<std::string,std::vector<Tensor>> rows;std::vector<Tensor> inert;
  for(const auto& s:states){
    auto stamps=host_ids(s.slots.at("times"));out.layout.times.insert(out.layout.times.end(),stamps.begin(),stamps.end());
    out.layout.offsets.push_back(out.layout.times.size());out.last_times.push_back(s.last_time);out.observations.push_back(s.observations);
    inert.push_back(s.value);
    for(const auto& name:{"keys","values","summaries","weights"})rows[name].push_back(s.slots.at(name));
  }
  out.inert=at::stack(inert);for(const auto& [name,values]:rows)out.fields[name]=at::cat(values);
  out.layout.validate();return out;
}
std::vector<State> KVState::states()const {
  std::vector<State> out;auto times=indices(layout.times,inert.device());
  for(Index b=0;b<layout.samples();++b){
    State s{inert[b],last_times[b],observations[b],{}};
    for(const auto& [name,t]:fields)s.slots[name]=t.slice(0,layout.offsets[b],layout.offsets[b+1]);
    s.slots["times"]=times.slice(0,layout.offsets[b],layout.offsets[b+1]);out.push_back(s);
  }
  return out;
}
void KVState::validate(const NodeWeights& w,const KVConfig& c)const {
  layout.validate();const Index n=layout.times.size(),b=layout.samples();
  require(last_times.size()==size_t(b)&&observations.size()==size_t(b),"KV clock/count domain differs");
  require(fields.size()==4&&inert.sizes()==at::IntArrayRef({b,c.width+1})&&inert.count_nonzero().item<Index>()==0,"invalid KV inert state");
  std::map<std::string,std::vector<Index>> shapes{{"keys",{n,c.width}},{"values",{n,c.width}},
      {"summaries",{n,c.summary_width}},{"weights",{n}}};
  for(const auto& [name,shape]:shapes){const auto& t=fields.at(name);
    require(t.sizes()==at::IntArrayRef(shape)&&t.device()==w.bias.device()&&t.scalar_type()==w.bias.scalar_type()
      &&at::isfinite(t).all().item<bool>(),"invalid KV raw column shape/metadata/value");}
  require(inert.device()==w.bias.device()&&inert.scalar_type()==w.bias.scalar_type(),"KV inert metadata differs");
  auto weights=fields.at("weights");require(!((weights<0)|(weights>1)).any().item<bool>(),"KV write weight outside [0,1]");
  for(Index i=0;i<b;++i){const auto a=layout.offsets[i],z=layout.offsets[i+1];
    require(last_times[i]>=-1&&observations[i]>=z-a&&(observations[i]==0)==(last_times[i]==-1)
      &&(!c.window||z-a<=c.window)&&(z==a||layout.times[z-1]<=last_times[i]),"invalid KV column clocks/counts");}
}
bool supported(const KVConfig& c){
  c.validate();return true;
}
namespace {
struct Preparation {Tensor ids;std::vector<Index> writes;};
Preparation prepare_inputs(const NodeWeights& w,const KVConfig& c,const Layout& layout,const Tensor& mass,const Tensor& u,const KVState& left){
  c.validate();layout.validate();left.validate(w,c);const Index n=layout.times.size();
  require(layout.samples()==left.layout.samples()&&u.sizes()==at::IntArrayRef({n,c.width})&&mass.sizes()==at::IntArrayRef({n}),"KV window dimensions differ");
  require(u.device()==w.bias.device()&&u.scalar_type()==w.bias.scalar_type()&&mass.device()==u.device()&&mass.scalar_type()==u.scalar_type()
    &&at::isfinite(u).all().item<bool>()&&at::isfinite(mass).all().item<bool>()&&!(mass<0).any().item<bool>(),"invalid KV window payload");
  for(Index b=0;b<layout.samples();++b)if(layout.offsets[b]<layout.offsets[b+1])
    require(layout.times[layout.offsets[b]]>left.last_times[b],"KV column window must advance logical time");
  auto ids=(mass.detach()>0).nonzero().flatten();auto writes=host_ids(ids);
  return {ids,std::move(writes)};
}
KVWindow assemble(const KVConfig& c,const Layout& layout,const Tensor& mass,const Tensor& u,
                  const KVState& left,const Preparation& preparation,const TensorFields& fresh){
  KVWindow out{layout,left.layout,mass,u,left.fields,left};
  const auto& writes=preparation.writes;if(writes.empty())return out;
  Layout bank;bank.offsets={0};std::vector<Index> perm;size_t next=0;
  for(Index b=0;b<layout.samples();++b){
    for(Index j=left.layout.offsets[b];j<left.layout.offsets[b+1];++j){perm.push_back(j);bank.times.push_back(left.layout.times[j]);}
    Index count=0;
    while(next<writes.size()&&writes[next]<layout.offsets[b+1]){
      perm.push_back(left.layout.times.size()+next);bank.times.push_back(layout.times[writes[next]]);++next;++count;}
    bank.offsets.push_back(bank.times.size());
    if(count){require(out.terminal.observations[b]<=std::numeric_limits<Index>::max()-count,"KV observation overflow");
      out.terminal.last_times[b]=bank.times.back();out.terminal.observations[b]+=count;}
  }
  auto order=indices(perm,u.device());out.bank_layout=bank;out.terminal.layout=bank;
  for(const auto& [name,t]:fresh)out.bank[name]=at::cat({left.fields.at(name),t}).index_select(0,order);
  out.terminal.fields=out.bank;
  if(c.window){Layout retained;retained.offsets={0};std::vector<Index> keep;
    for(Index b=0;b<layout.samples();++b){
      for(Index j=bank.offsets[b];j<bank.offsets[b+1];++j)
        if(out.terminal.observations[b]==left.observations[b]||out.terminal.last_times[b]-bank.times[j]<c.window){keep.push_back(j);retained.times.push_back(bank.times[j]);}
      retained.offsets.push_back(retained.times.size());}
    auto ids=indices(keep,u.device());out.terminal.layout=retained;
    for(const auto& [name,t]:out.bank)out.terminal.fields[name]=t.index_select(0,ids);
  }
  return out;
}
TensorFields projected(const NodeWeights& w,const KVConfig& c,const Tensor& mass,const Tensor& u,const Preparation& p){
  if(p.writes.empty())return {};
  auto x=normalize(u.index_select(0,p.ids),c.epsilon);
  auto kv=project(x,weight(w,"kernel.base"))+project(project(x,weight(w,"kernel.adapter_v")),weight(w,"kernel.adapter_u"));
  auto split=kv.chunk(2,-1);
  return {{"keys",split[0]},{"values",split[1]},{"summaries",project(x,weight(w,"kernel.summary"))},
          {"weights",mass.index_select(0,p.ids).clamp_max(1)}};
}
long double group_bytes(const KVConfig& c,Index rows,Index element_size){
  // Sum four block scopes: two weight copies, three inputs, two outputs,
  // plus gathered/normalized content and the KV sum. No R-wise outer products.
  long double d=c.width,a=c.adapter_rank,s=c.summary_width;
  return (2*(2*d*d+3*d*a+d*s)+rows*(25*d+5*a+2*s))*std::max<Index>(element_size,4)+16.L*rows;
}
}
KVWindow prepare(const NodeWeights& w,const KVConfig& c,const Layout& layout,const Tensor& mass,const Tensor& u,const KVState& left){
  auto p=prepare_inputs(w,c,layout,mass,u,left);
  return assemble(c,layout,mass,u,left,p,projected(w,c,mass,u,p));
}
std::vector<KVWindow> prepare_many(const std::vector<KVPrepareRequest>& requests,Index max_workspace_bytes){
  require(max_workspace_bytes>0,"KV preparation group budget must be positive");
  using Key=std::tuple<int,int,int,Index,Index,Index,Index,double>;
  std::map<Key,std::vector<size_t>> groups;
  std::vector<Preparation> prepared;std::vector<KVWindow> result(requests.size());prepared.reserve(requests.size());
  for(size_t i=0;i<requests.size();++i){const auto& q=requests[i];
    require(q.weights!=nullptr,"KV preparation requires a live parameter owner");
    prepared.push_back(prepare_inputs(*q.weights,q.config,q.layout,q.mass,q.content,q.left));
    const auto rows=static_cast<Index>(prepared.back().writes.size());
    if(!rows){result[i]=assemble(q.config,q.layout,q.mass,q.content,q.left,prepared.back(),{});continue;}
    const auto device=q.content.device();const auto& c=q.config;
    const bool compact=q.content.scalar_type()==at::kFloat||q.content.scalar_type()==at::kDouble;
    groups[{int(device.type()),device.index(),int(q.content.scalar_type()),compact?0:rows,c.width,c.adapter_rank,c.summary_width,c.epsilon}].push_back(i);
  }
  auto execute=[&](const std::vector<size_t>& ids){
    if(ids.size()==1){const auto i=ids.front();const auto& q=requests[i];
      result[i]=assemble(q.config,q.layout,q.mass,q.content,q.left,prepared[i],projected(*q.weights,q.config,q.mass,q.content,prepared[i]));return;}
    const auto dtype=requests[ids[0]].content.scalar_type();
    if(dtype==at::kFloat||dtype==at::kDouble){
      std::vector<tensor::WriteRequest> rows;
      for(auto i:ids){const auto& q=requests[i];rows.push_back({q.weights,q.mass,q.content,&prepared[i].writes});}
      auto fresh=tensor::prepare(rows,requests[ids[0]].config.epsilon);
      for(size_t j=0;j<ids.size();++j){const auto i=ids[j];const auto& q=requests[i];
        result[i]=assemble(q.config,q.layout,q.mass,q.content,q.left,prepared[i],fresh[j]);}
      return;
    }
    std::vector<Tensor> xs,base,adapter_v,adapter_u,summary;
    for(auto i:ids){const auto& q=requests[i];const auto& w=*q.weights;
      xs.push_back(normalize(q.content.index_select(0,prepared[i].ids),q.config.epsilon));
      base.push_back(weight(w,"kernel.base"));adapter_v.push_back(weight(w,"kernel.adapter_v"));
      adapter_u.push_back(weight(w,"kernel.adapter_u"));summary.push_back(weight(w,"kernel.summary"));}
    auto bases=linear_blocks(xs,base),low=linear_blocks(xs,adapter_v);
    auto updates=linear_blocks(low,adapter_u),summaries=linear_blocks(xs,summary);
    for(size_t j=0;j<ids.size();++j){const auto i=ids[j];const auto& q=requests[i];
      auto split=(bases[j]+updates[j]).chunk(2,-1);
      TensorFields fresh{{"keys",split[0]},{"values",split[1]},{"summaries",summaries[j]},
                         {"weights",q.mass.index_select(0,prepared[i].ids).clamp_max(1)}};
      result[i]=assemble(q.config,q.layout,q.mass,q.content,q.left,prepared[i],fresh);}
  };
  for(const auto& [key,ids]:groups){std::vector<size_t> batch;Index longest=0;long double originals=0;
    for(auto i:ids){const auto& q=requests[i];const auto candidate=std::max(longest,static_cast<Index>(prepared[i].writes.size()));
      const auto count=batch.size()+1;const auto source_rows=originals+q.layout.times.size();
      auto cost=count*group_bytes(q.config,candidate,q.content.element_size())
        +source_rows*(8+2*(q.config.width+1)*std::max<Index>(4,q.content.element_size()));
      if(!batch.empty()&&cost>max_workspace_bytes){execute(batch);batch.clear();longest=0;originals=0;}
      batch.push_back(i);longest=std::max(longest,static_cast<Index>(prepared[i].writes.size()));originals+=q.layout.times.size();
    }
    if(!batch.empty())execute(batch);
  }
  return result;
}
}
