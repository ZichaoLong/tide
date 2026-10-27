#include "weighted_recurrent_internal.h"
#include <cmath>
namespace tide::device_online::recurrent_detail {
namespace {
struct Algebra {
  WeightedRecurrentTape& t;WeightedRowGraph& g;const weighted::RecurrentConfig& c;const WeightedRecurrentParameters& w;
  DeviceProgram& program;const at::Tensor& times;const at::Tensor& error;int64_t budget;size_t u,x,H,Z,W,one;
  Algebra(WeightedRecurrentTape& tape,const WeightedRecurrentParameters& weights,DeviceProgram& p,const at::Tensor& time,const at::Tensor& e,int64_t b)
    :t(tape),g(*t.graph),c(t.config),w(weights),program(p),times(time),error(e),budget(b){
    u=t.inputs.at("content");x=g.norm(u,c.epsilon);H=t.inputs.at("H");Z=t.inputs.at("Z");W=t.inputs.at("W");one=g.constant(1,1);
  }
  size_t linear(const std::string& role,size_t input){return g.projection(input,{w.at(role+".matrix"),w.count(role+".bias")?w.at(role+".bias"):at::Tensor{}},role);}
  size_t minimum_one(size_t v){return g.clamp_max(v,1);}
  size_t maximum_one(size_t v){return g.scale(g.clamp_max(g.scale(v,-1),-1),-1);}
  size_t negative(size_t v){return g.scale(v,-1);}
  size_t slice(size_t v,int64_t start,int64_t n){return g.slice(v,start,n);}
  size_t read(size_t matrix,size_t query){return g.matmul(matrix,query,c.value_width,c.key_width,1);}
  size_t matrix(int64_t h){return slice(H,h*c.value_width*c.key_width,c.value_width*c.key_width);}
  size_t outer(size_t value,size_t key){return g.matmul(value,key,c.value_width,1,c.key_width);}
  size_t l2(size_t v){return g.scale(g.norm(v,c.qk_epsilon/c.key_width),1/std::sqrt(c.key_width));}
  struct Features {size_t raw=0,gate=0,delta=0;std::vector<size_t> query,key,value;};
  Features features(){
    Features f;auto projected=linear("projection",x);const auto hv=c.heads*c.value_width;
    if(c.memory_family=="gated_delta")f.raw=projected;
    else {f.gate=slice(projected,0,hv);f.raw=slice(projected,hv,c.conv_width());
      f.delta=g.softplus(g.add(slice(projected,hv+c.conv_width(),c.heads),linear("delta_bias",one)));}
    auto conv=g.silu(g.convolution(f.raw,t.inputs.at("conv_values"),t.snapshot.layout,t.snapshot.state.integer,times,
      {w.at("conv.weight"),w.at("conv.bias")},c.conv_stride,"conv"));
    for(int64_t h=0;h<c.heads;++h){if(c.memory_family=="gated_delta"){
        const auto off=h*(2*c.key_width+c.value_width);
        f.query.push_back(g.scale(l2(slice(conv,off,c.key_width)),1/std::sqrt(c.key_width)));
        f.key.push_back(l2(slice(conv,off+c.key_width,c.key_width)));f.value.push_back(slice(conv,off+2*c.key_width,c.value_width));
      }else {const auto group=h/(c.heads/c.groups);
        f.value.push_back(slice(conv,h*c.value_width,c.value_width));f.key.push_back(slice(conv,hv+group*c.key_width,c.key_width));
        f.query.push_back(slice(conv,hv+c.groups*c.key_width+group*c.key_width,c.key_width));}}
    return f;
  }
  void update(){
    const auto mass=minimum_one(t.inputs.at("mass")),summary=linear("summary",x);std::vector<size_t> matrices,zs,ws;
    if(c.basic()){
      auto key=linear("key",x);key=g.divide(key,g.sqrt(maximum_one(g.sum(g.product(key,key)))));
      auto value=linear("value",x),retain=g.exp(negative(g.product(mass,g.softplus(linear("retain",x)))));
      const auto beta=c.memory_family=="delta"?g.product(mass,g.sigmoid(linear("write",x))):mass;
      for(int h=0;h<2;++h){auto retained=g.product(retain,matrix(h)),residual=value;
        if(c.memory_family=="delta")residual=g.add(value,negative(read(retained,key)));
        matrices.push_back(g.add(retained,g.product(beta,outer(residual,key))));}
    }else{
      auto f=features();auto delta=c.memory_family=="gated_delta"?g.softplus(linear("delta",x)):f.delta;
      auto step=g.product(mass,delta),retain=g.exp(negative(g.product(g.exp(linear("log_rate",one)),step)));
      auto beta=c.memory_family=="gated_delta"?g.product(mass,g.sigmoid(linear("write",x))):step;
      for(int64_t h=0;h<c.heads;++h){auto retained=g.product(slice(retain,h,1),matrix(h)),residual=f.value[h];
        if(c.memory_family=="gated_delta")residual=g.add(residual,negative(read(retained,f.key[h])));
        matrices.push_back(g.add(retained,g.product(slice(beta,h,1),outer(residual,f.key[h]))));}
      t.outputs["conv_values"]=f.raw;
    }
    for(int a=0;a<2;++a){zs.push_back(g.add(slice(Z,a*c.summary_width,c.summary_width),g.product(mass,summary)));ws.push_back(g.add(slice(W,a,1),mass));}
    t.outputs["H"]=g.cat(matrices);t.outputs["Z"]=g.cat(zs);t.outputs["W"]=g.cat(ws);
  }
  void memory(){
    auto short_w=slice(W,0,1),long_w=slice(W,1,1);size_t short_m,long_m;
    auto summary=[&](int a,size_t weight){return g.divide(slice(Z,a*c.summary_width,c.summary_width),maximum_one(weight));};
    if(c.memory_read=="sum"){short_m=summary(0,short_w);long_m=summary(1,long_w);}
    else if(c.basic()){
      auto query=linear("read.query",x);short_m=linear("read.out.0",read(matrix(0),query));long_m=linear("read.out.1",read(matrix(1),query));
    }else{
      short_m=summary(0,short_w);auto f=features();std::vector<size_t> reads;for(int64_t h=0;h<c.heads;++h)reads.push_back(read(matrix(h),f.query[h]));
      long_m=linear("read.out",g.cat(reads));
    }
    t.outputs={{"short_memory",short_m},{"long_memory",long_m},{"short_total",short_w},{"long_total",long_w}};
  }
  void full(){
    size_t mix;if(c.basic()){
      auto query=linear("full.query",x);mix=g.add(linear("full.out.0",read(matrix(0),query)),linear("full.out.1",read(matrix(1),query)));
    }else{
      auto f=features();std::vector<size_t> reads;auto rho=linear("full.rho",one);
      if(c.memory_family=="gated_delta"){
        for(int64_t h=0;h<c.heads;++h)reads.push_back(g.norm(read(matrix(h),f.query[h]),c.output_epsilon));
        auto gated=g.product(g.product(g.cat(reads),rho),g.silu(linear("full.output_gate",x)));mix=linear("full.out",gated);
      }else{
        auto direct=linear("full.direct",one);
        for(int64_t h=0;h<c.heads;++h)reads.push_back(g.add(read(matrix(h),f.query[h]),g.product(slice(direct,h,1),f.value[h])));
        auto gated=g.product(g.cat(reads),g.silu(f.gate));mix=linear("full.out",g.product(rho,g.norm(gated,c.output_epsilon)));
      }
    }
    auto residual=g.add(u,g.scale(mix,c.alpha_mem)),normalized=g.norm(residual,c.epsilon);
    auto ff=linear("full.down",g.product(g.silu(linear("full.gate",normalized)),linear("full.up",normalized)));
    t.outputs["G"]=g.add(residual,g.scale(ff,c.alpha_ff));
  }
};
}
void recurrent_algebra(WeightedRecurrentTape& t,const WeightedRecurrentParameters& w,DeviceProgram& p,
    const at::Tensor& times,const at::Tensor& error,int64_t budget){
  Algebra a(t,w,p,times,error,budget);
  if(t.action==WeightedRecurrentAction::Update)a.update();else if(t.action==WeightedRecurrentAction::Memory)a.memory();else a.full();
}
}
