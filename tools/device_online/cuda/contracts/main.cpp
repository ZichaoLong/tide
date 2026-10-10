#define TIDE_KERNEL_CPU
#include "kernel_operator.h"
#include "../../control_flow.h"
#include "device_launch_tide_closure.h"
#include <iostream>
#include <vector>
#include <limits>

namespace {
void require(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
template<class F> void reject(F fn){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}require(rejected,"invalid source contract accepted");}
void primitives() {
  using namespace AscendC;
  tide_cuda::reset_arena();TPipe pipe;TBuf<QuePosition::VECCALC> data,rounding,bits;
  pipe.InitBuffer(data,256*4);pipe.InitBuffer(rounding,256*2);pipe.InitBuffer(bits,32);
  auto values=data.Get<float>();auto half_values=rounding.Get<half>();
  values.SetValue(0,1.0006f);values.SetValue(1,-0.f);values.SetValue(2,65520.f);
  Cast(half_values,values,RoundMode::CAST_RINT,3);Cast(values,half_values,RoundMode::CAST_NONE,3);
  require(values.GetValue(0)==1.0009765625f&&std::signbit(values.GetValue(1))&&std::isinf(values.GetValue(2)),"half rounding contract");
  float input[3]={2,3,4};GlobalTensor<float> global;global.SetGlobalBuffer(input);
  DataCopyPad(values,global,{1,12,0,0,0},{true,0,5,0.f});
  require(values.GetValue(2)==4&&values.GetValue(7)==0,"tail copy/padding contract");
  require(input[0]==2&&input[2]==4,"copy mutated input");
  values.SetValue(3,std::numeric_limits<float>::quiet_NaN());
  Compares(bits.Get<uint8_t>(),values,3.f,CMPMODE::LE,4);
  require(bits.Get<uint8_t>().GetValue(0)==3,"comparison bits or NaN ordering");
  require(GetScalarBitcodeValue<uint32_t,float>(0x80000000u)==0&&std::signbit(GetScalarBitcodeValue<uint32_t,float>(0x80000000u)),"bitcast sign");
  reject([&]{values.GetValue(256);});reject([&]{TPipe p;TBuf<QuePosition::VECCALC> x;p.InitBuffer(x,32768);});
}
void closure() {
  using I=int64_t;const I t=(I(1)<<53)+17;
  I coordinates[]={0,0,t,0,0,0, 0,0,t+1,0,0,1, 0,1,t+3,0,0,2};
  uint8_t valid[]={1,1,1};I owners[]={0,1},distance[]={1,2,3,1},work[5],stop=t+20;
  int32_t ready[3]={},branch=0,error=0;
  auto call=[&](int64_t greedy){return TIDE_LAUNCH_KERNEL(tide_closure)(1,nullptr,coordinates,valid,owners,distance,work,&stop,ready,&branch,&error,3,2,2,1,greedy);};
  require(call(1)==0&&error==0&&branch==1&&ready[0]==1&&ready[1]==0&&ready[2]==0,"positive-delay exact greedy closure");
  require(work[0]==t&&work[1]==t+3&&work[2]==t+1&&work[3]==t+2,"int64 closure bounds");
  valid[0]=0;require(call(0)==0&&error==0&&ready[1]==1&&ready[2]==0,"online next closure after consumption");
  stop=t;require(call(0)==0&&branch==0&&!ready[1]&&!ready[2],"window seal closure");
  stop=t+20;distance[0]=0;call(1);require(error==2&&branch==0&&!ready[0]&&!ready[1]&&!ready[2],"invalid delay atomic refusal");
}
void control() {
  using namespace tide::device_online;using C=ControlInstruction;
  std::vector<C> code={{C::Mark,0,{}},{},{C::Branch,0,{2,1}},{C::Mark,1,{}},{},{C::Branch,0,{0}},{C::Mark,2,{}}};
  const auto blocks=lower_control(code,3);
  require(blocks.size()==3&&blocks[0].targets==std::vector<int32_t>{2,1}&&blocks[1].targets==std::vector<int32_t>{0},"conditional CFG edges");
  for(int limit:{0,1,5}) {
    int pc=0,count=0,steps=0;
    while(pc<int(blocks.size())) {
      require(++steps<32,"CFG failed to exit");auto& b=blocks[pc];
      for(auto op:b.operations)if(op==4)++count;
      auto branch=b.branch==2?(count<limit?1:0):0;pc=b.targets[branch];
    }
    require(count==limit,"device-loop lowering semantic simulation");
  }
  reject([&]{lower_control(code,4);});code.push_back({C::Mark,0,{}});reject([&]{lower_control(code,3);});
}
}
int main(){try{primitives();closure();control();std::cout<<"CUDA source/primitive/control contracts passed; GPU execution unverified\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
