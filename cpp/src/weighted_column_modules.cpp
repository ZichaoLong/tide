#include "weighted_column_modules.h"
namespace tide::weighted::columns {
namespace {
template<class C> using StateFor=std::conditional_t<std::is_same_v<C,KVConfig>,KVState,RecurrentState>;
template<class C> using WindowFor=std::conditional_t<std::is_same_v<C,KVConfig>,KVWindow,RecurrentWindow>;
}
LocalState initial_columns(const NodeWeights& w,const MemoryConfiguration& config,Index samples){
  std::vector<State> values(samples,w.kernel->initial(w));
  return std::visit([&](const auto& c)->LocalState{using C=std::decay_t<decltype(c)>;
    return StateFor<C>::from_states(values);},config);
}
void validate_columns(const NodeWeights& w,const MemoryConfiguration& config,const LocalState& state){
  std::visit([&](const auto& c){using C=std::decay_t<decltype(c)>;
    detail::require(std::holds_alternative<StateFor<C>>(state),"local state type differs from configured memory");
    std::get<StateFor<C>>(state).validate(w,c);},config);
}
LocalWindow prepare_columns(const NodeWeights& w,const MemoryConfiguration& config,const Layout& layout,
                            const Tensor& mass,const Tensor& u,const LocalState& state){
  return std::visit([&](const auto& c)->LocalWindow{using C=std::decay_t<decltype(c)>;
    return prepare(w,c,layout,mass,u,std::get<StateFor<C>>(state));},config);
}
TensorFields read_columns(const NodeWeights& w,const NodeBinding& b,const LocalWindow& window,
                          const std::string& mode,const LocalState& initial){
  return std::visit([&](const auto& c){using C=std::decay_t<decltype(c)>;
    return read(w,c,b.regional,std::get<WindowFor<C>>(window),mode,std::get<StateFor<C>>(initial));},b.memory);
}
Message full_columns(const NodeWeights& w,const MemoryConfiguration& config,const LocalWindow& window,const Tensor& controls){
  return std::visit([&](const auto& c){using C=std::decay_t<decltype(c)>;
    return full(w,c,std::get<WindowFor<C>>(window),controls);},config);
}
LocalState terminal_columns(const LocalWindow& window){return std::visit([](const auto& w)->LocalState{return w.terminal;},window);}
long double workspace_bytes(const MemoryConfiguration& config,const Layout& domain,const LocalState& state,Index element_size){
  return std::visit([&](const auto& c)->long double{using C=std::decay_t<decltype(c)>;
    long double rows=domain.times.size();
    if constexpr(std::is_same_v<C,KVConfig>){
      Index queries=0,past=0;const auto& old=std::get<KVState>(state).layout;
      for(Index owner=0;owner<domain.samples();++owner){
        queries=std::max(queries,domain.offsets[owner+1]-domain.offsets[owner]);
        past=std::max(past,old.offsets[owner+1]-old.offsets[owner]);}
      long double pairs=static_cast<long double>(domain.samples())*queries*(static_cast<long double>(past)+queries);
      long double elements=pairs*(8+3*c.heads);
      bool rotary=!c.read_frequencies.empty();for(const auto& f:c.full_frequencies)rotary|=!f.empty();
      if(rotary)elements+=8*pairs*(c.width+c.summary_width);
      return (elements+rows*(4*c.width+c.summary_width))*element_size;
    }else{
      Index length=0;for(Index b=0;b<domain.samples();++b)length=std::max(length,domain.offsets[b+1]-domain.offsets[b]);
      long double padded=static_cast<long double>(domain.samples())*length;
      long double elements=8*padded*c.matrices()*(static_cast<long double>(c.key_width)*c.key_width+static_cast<long double>(c.key_width)*c.value_width)
                          +padded*(8*c.width+8*c.summary_width);
      long double metadata=0;
      if(!c.basic()){
        long double bank=std::get<RecurrentState>(state).cache_layout.times.size()+rows,taps=c.conv_size-1;
        elements+=4*(bank+rows*c.conv_size+bank*taps)*c.conv_width();
        metadata=16*(rows+bank)*taps; // host/device int64 forward and transpose maps
      }
      return elements*element_size+metadata;
    }
  },config);
}
}
