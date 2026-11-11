// Raw recurrent state/scan: exact time and convolution slots, typed H/Z/W.
#include "weighted_column_recurrent.h"
#include "weighted_packed.h"
namespace tide::weighted::columns {
using detail::require;
RecurrentState RecurrentState::from_states(const std::vector<State>& states){
  require(!states.empty(),"recurrent columns require samples");RecurrentState out;
  std::map<std::string,std::vector<Tensor>> fields;std::vector<Tensor> inert,cache;
  const bool has_cache=states[0].slots.count("conv_times");if(has_cache)out.cache_layout.offsets={0};
  for(const auto& s:states){
    const auto& time=s.slots.at("time");
    require(time.scalar_type()==at::kLong&&time.dim()==0&&time.item<Index>()==std::max(Index{0},s.last_time),"invalid raw recurrent clock");
    for(const auto& n:{"H","Z","W"})fields[n].push_back(s.slots.at(n));inert.push_back(s.value);
    out.last_times.push_back(s.last_time);out.observations.push_back(s.observations);
    require(bool(s.slots.count("conv_times"))==has_cache,"inconsistent recurrent cache type");
    if(has_cache){auto times=host_ids(s.slots.at("conv_times"));
      out.cache_layout.times.insert(out.cache_layout.times.end(),times.begin(),times.end());
      out.cache_layout.offsets.push_back(out.cache_layout.times.size());cache.push_back(s.slots.at("conv_values"));}
  }
  for(const auto& [n,rows]:fields)out.fields[n]=at::stack(rows);out.inert=at::stack(inert);
  if(has_cache)out.cache_values=at::cat(cache);return out;
}
std::vector<State> RecurrentState::states()const{
  std::vector<State> out;
  for(size_t b=0;b<last_times.size();++b){
    State s;s.value=inert[b];s.last_time=last_times[b];s.observations=observations[b];
    for(const auto& [n,t]:fields)s.slots[n]=t[b];
    s.slots["time"]=at::scalar_tensor(std::max(Index{0},s.last_time),inert.options().dtype(at::kLong));
    if(!cache_layout.offsets.empty()){
      const auto a=cache_layout.offsets[b],z=cache_layout.offsets[b+1];
      s.slots["conv_times"]=indices({cache_layout.times.begin()+a,cache_layout.times.begin()+z},inert.device());
      s.slots["conv_values"]=cache_values.slice(0,a,z);
    }out.push_back(s);
  }return out;
}
void RecurrentState::validate(const NodeWeights& w,const RecurrentConfig& c)const{
  c.validate();const auto b=static_cast<Index>(last_times.size());require(b>0&&observations.size()==last_times.size(),"invalid recurrent sample domain");
  const std::map<std::string,std::vector<Index>> shapes{{"H",{b,c.matrices(),c.value_width,c.key_width}},
      {"Z",{b,2,c.summary_width}},{"W",{b,2}}};
  require(fields.size()==3,"invalid recurrent column fields");
  for(const auto& [n,shape]:shapes){const auto& x=fields.at(n);
    require(x.sizes()==at::IntArrayRef(shape)&&x.scalar_type()==w.bias.scalar_type()&&x.device()==w.bias.device()
      &&at::isfinite(x).all().item<bool>(),"invalid recurrent state column");}
  require(inert.sizes()==at::IntArrayRef({b,c.width+1})&&inert.device()==w.bias.device()&&inert.scalar_type()==w.bias.scalar_type()
    &&(inert==0).all().item<bool>()&&!(fields.at("W")<0).any().item<bool>(),"invalid recurrent inert state/count");
  for(Index i=0;i<b;++i)require(last_times[i]>=-1&&observations[i]>=0&&((last_times[i]==-1)==(observations[i]==0)),"invalid recurrent clock/count");
  if(c.basic()){require(cache_layout.offsets.empty()&&!cache_values.defined(),"basic recurrent state has no cache");return;}
  cache_layout.validate();require(cache_layout.samples()==b,"recurrent cache sample domain differs");
  require(cache_values.defined()&&cache_values.sizes()==at::IntArrayRef({static_cast<Index>(cache_layout.times.size()),c.conv_width()})
    &&cache_values.scalar_type()==w.bias.scalar_type()&&cache_values.device()==w.bias.device()
    &&at::isfinite(cache_values).all().item<bool>(),"invalid recurrent cache columns");
  const Index horizon=(c.conv_size-1)*c.conv_stride;
  for(Index i=0;i<b;++i){auto a=cache_layout.offsets[i],z=cache_layout.offsets[i+1];
    require(z-a<=std::min(observations[i],horizon+1)&&(z==a||cache_layout.times[z-1]<=last_times[i]),"invalid raw recurrent cache clock/count");}
}
namespace {
Tensor scan(const Layout& layout,const Tensor& a,const Tensor& b,const Tensor& initial,bool matrix=false){
  auto [ids,reverse]=padded(layout,b.device());auto pa=gather(a,ids),pb=gather(b,ids);
  auto shape=ids.sizes().vec();for(Index i=1;i<a.dim();++i)shape.push_back(1);
  auto mask=(ids<static_cast<Index>(layout.times.size())).reshape(shape);
  auto identity=matrix?at::eye(a.size(-1),a.options()):at::ones({},a.options());
  pa=at::where(mask,pa,identity);
  auto result=matrix?packed::matrix_scan(pa.transpose(0,1),pb.transpose(0,1),initial):affine_scan(pa.transpose(0,1),pb.transpose(0,1),initial);
  return result.transpose(0,1).flatten(0,1).index_select(0,reverse);
}
}
RecurrentWindow prepare(const NodeWeights& w,const RecurrentConfig& c,const Layout& layout,const Tensor& mass,
                        const Tensor& content,const RecurrentState& left){
  layout.validate();left.validate(w,c);const auto n=static_cast<Index>(layout.times.size());
  require(layout.samples()==static_cast<Index>(left.last_times.size())&&mass.sizes()==at::IntArrayRef({n})
    &&content.sizes()==at::IntArrayRef({n,c.width}),"invalid recurrent column input shapes");
  for(const auto& t:{mass,content})require(t.device()==w.bias.device()&&t.scalar_type()==w.bias.scalar_type()
    &&at::isfinite(t).all().item<bool>(),"invalid recurrent column input metadata/value");
  require(!(mass<0).any().item<bool>(),"negative recurrent column mass");
  for(Index b=0;b<layout.samples();++b)if(layout.offsets[b+1]>layout.offsets[b])
    require(layout.times[layout.offsets[b]]>left.last_times[b],"recurrent window must advance logical time");
  auto ids=(mass.detach()>0).nonzero().flatten();auto rows=host_ids(ids);auto writes=layout.select(rows);
  RecurrentWindow result;result.layout=layout;result.write_layout=writes;result.mass=mass;result.content=content;
  result.bank=left.fields;result.cache_layout=left.cache_layout;result.cache_values=left.cache_values;result.terminal=left;
  for(auto time:left.last_times)result.times.push_back(std::max(Index{0},time));
  if(rows.empty())return result;
  auto u=content.index_select(0,ids),gain=mass.index_select(0,ids).clamp_max(1),x=normalize(u,c.epsilon);
  std::vector<Index> gaps,ends;auto& terminal=result.terminal;
  for(Index owner=0;owner<layout.samples();++owner){
    auto a=writes.offsets[owner],z=writes.offsets[owner+1],previous=std::max(Index{0},left.last_times[owner]);
    for(Index j=a;j<z;++j){gaps.push_back(writes.times[j]-previous);previous=writes.times[j];}
    ends.push_back(z>a?layout.samples()+z-1:owner);
    if(z>a){require(left.observations[owner]<=std::numeric_limits<Index>::max()-(z-a),"recurrent observation overflow");
      terminal.observations[owner]+=z-a;terminal.last_times[owner]=writes.times[z-1];}
  }
  auto cache=rec::cache_bank(w,c,writes,x,gain,left);result.cache_layout=cache.first;result.cache_values=cache.second;
  auto gap=indices(gaps,content.device());auto coef=rec::coefficients(w,c,writes,u,x,gain,gap,cache.first,cache.second);
  auto decay=detail::passive_decay(gap.unsqueeze(-1),{c.lambda_short,0.},x);
  auto z=gain.unsqueeze(-1)*project(normalize(u,c.epsilon),detail::weight(w,"kernel.summary"));
  TensorFields fresh{{"H",scan(writes,coef.a,coef.b,left.fields.at("H"),coef.matrix)},
    {"Z",scan(writes,decay.unsqueeze(-1),z.unsqueeze(1).expand({-1,2,-1}),left.fields.at("Z"))},
    {"W",scan(writes,decay,gain.unsqueeze(-1).expand({-1,2}),left.fields.at("W"))}};
  for(const auto& [name,t]:fresh){result.bank[name]=at::cat({left.fields.at(name),t});
    terminal.fields[name]=result.bank[name].index_select(0,indices(ends,content.device()));}
  result.times.insert(result.times.end(),writes.times.begin(),writes.times.end());
  if(!c.basic()){
    const Index horizon=(c.conv_size-1)*c.conv_stride;std::vector<Index> keep;terminal.cache_layout={{0},{}};
    for(Index owner=0;owner<layout.samples();++owner){
      const bool changed=terminal.observations[owner]!=left.observations[owner];
      for(Index j=cache.first.offsets[owner];j<cache.first.offsets[owner+1];++j)
        if(!changed||terminal.last_times[owner]-cache.first.times[j]<=horizon){keep.push_back(j);terminal.cache_layout.times.push_back(cache.first.times[j]);}
      terminal.cache_layout.offsets.push_back(keep.size());
    }
    terminal.cache_values=cache.second.index_select(0,indices(keep,content.device()));
  }return result;
}
}
