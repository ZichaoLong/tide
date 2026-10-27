#pragma once
#include "kernel_operator.h"
#include "../weighted_allocation_layout.h"
namespace tide_weighted_device {
using I=int64_t;
__aicore__ inline bool finite(float x){return (AscendC::GetScalarBitcodeValue<float,uint32_t>(x)&0x7f800000)!=0x7f800000;}
__aicore__ inline float magnitude(float x){return x<0?-x:x;}
__aicore__ inline float minimum(float a,float b){return a<b?a:b;}
__aicore__ inline float maximum(float a,float b){return a>b?a:b;}
__aicore__ inline float inf(){return AscendC::GetScalarBitcodeValue<uint32_t,float>(0x7f800000u);}
// Vendor vector elementary functions on a bounded scalar lane. Ragged active
// faces and bisection decisions stay on device. This is a correctness kernel;
// dense projections/normalization remain separate batched vendor operations.
class Math {
 public:
  __aicore__ inline void init(){pipe.InitBuffer(storage,32);}
  __aicore__ inline float unary(float x,bool logarithm){
    auto lane=storage.Get<float>();lane.SetValue(0,x);
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
    if(logarithm)AscendC::Ln(lane,lane,1);else AscendC::Exp(lane,lane,1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    return lane.GetValue(0);
  }
  __aicore__ inline float log(float x){return x==0?-inf():unary(x,true);}
  __aicore__ inline float exp(float x){return unary(x,false);}
  __aicore__ inline float log1p(float x){
    if(magnitude(x)<.015625f)return x*(1+x*(-.5f+x*(1.f/3+x*(-.25f+x*(.2f-x/6)))));
    return log(1+x);
  }
  __aicore__ inline float expm1(float x){
    if(magnitude(x)<.015625f)return x*(1+x*(.5f+x*(1.f/6+x*(1.f/24+x*(1.f/120+x/720)))));
    return exp(x)-1;
  }
  __aicore__ inline float sigmoid(float x){const float e=exp(-magnitude(x));return x>=0?1/(1+e):e/(1+e);}
 private:
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> storage;
};
class Response {
 public:
  I kind;float alpha;Math math;
  __aicore__ inline void init(I k,float a){kind=k;alpha=a;math.init();}
  __aicore__ inline float value(float x,float mu,float tau){
    using namespace tide::device_online::weighted_allocation;
    if(mu==0)return 0;
    const float z=x/tau;
    if(kind==Quad)return minimum(maximum(z,0.f),mu);
    if(kind==Bernoulli)return mu*math.sigmoid(z);
    const float log_cap=math.log(mu),arg=(alpha-1)*(z-1/alpha);
    const float log_value=alpha==1?z-1:(arg<=-1?-inf():math.log1p(arg)/(alpha-1));
    return log_value>=log_cap?mu:math.exp(log_value);
  }
  __aicore__ inline float inverse(float fraction,float mu,float tau){
    using namespace tide::device_online::weighted_allocation;
    if(kind==Quad)return tau*fraction*mu;
    if(kind==Bernoulli)return tau*(math.log(fraction)-math.log1p(-fraction));
    const float log_c=math.log(fraction)+math.log(mu);
    return alpha==1?tau*(1+log_c):tau*(math.expm1((alpha-1)*log_c)/(alpha-1)+1/alpha);
  }
  __aicore__ inline bool support(float x,float mu,float tau){
    using namespace tide::device_online::weighted_allocation;
    if(kind==Bernoulli||(kind==Entropy&&alpha==1))return mu>0;
    return mu>0&&(kind==Quad?x>0:x>-tau/(alpha*(alpha-1)));
  }
  __aicore__ inline void slopes(float x,float mu,float tau,float c,__gm__ float* out){
    using namespace tide::device_online::weighted_allocation;
    float a=0,e=0;
    if(kind==Bernoulli){const float p=math.sigmoid(x/tau);a=mu*p*math.sigmoid(-x/tau)/tau;e=mu>0?p:0;}
    else {if(c>0&&c<mu)a=kind==Quad?1/tau:math.exp((2-alpha)*math.log(c))/tau;e=c==mu&&mu>0?1:0;}
    out[Score]=a;out[Cap]=e;out[Tau]=a==0?0:-a*(x/tau);
  }
};
}
