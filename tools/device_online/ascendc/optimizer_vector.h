#pragma once
#include "kernel_operator.h"
namespace tide_device {
class OptimizerVector {
 public:
  __aicore__ inline void init(){pipe.InitBuffer(input,1,1024);pipe.InitBuffer(output,1,1024);pipe.InitBuffer(storage,8*1024);
    pipe.InitBuffer(finite_bits,32);}
  __aicore__ inline AscendC::LocalTensor<float> at(int index){return storage.Get<float>()[index*256];}
  __aicore__ inline void load(AscendC::LocalTensor<float> to,__gm__ float* from,int64_t offset,uint32_t count){
    AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(from);auto x=input.AllocTensor<float>();
    AscendC::DataCopyPad(x,gm[offset],AscendC::DataCopyExtParams{1,count*4,0,0,0},
      AscendC::DataCopyPadExtParams<float>{true,0,uint8_t((8-count%8)%8),0});
    input.EnQue(x);x=input.DeQue<float>();AscendC::Muls(to,x,1.f,count);AscendC::PipeBarrier<PIPE_V>();input.FreeTensor(x);
  }
  __aicore__ inline void save(AscendC::LocalTensor<float> from,__gm__ float* to,int64_t offset,uint32_t count){
    AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(to);auto x=output.AllocTensor<float>();AscendC::Muls(x,from,1.f,count);
    output.EnQue(x);x=output.DeQue<float>();AscendC::DataCopyPad(gm[offset],x,AscendC::DataCopyExtParams{1,count*4,0,0,0});output.FreeTensor(x);
  }
  __aicore__ inline bool finite(AscendC::LocalTensor<float> x,uint32_t count){
    // A finite float has abs(x) <= FLT_MAX; ordered comparison rejects both
    // NaN and infinity. The eighth value tile is private scratch. Inspect the
    // packed comparison bits, not every numerical lane; mask unused tail bits.
    auto magnitude=at(7);auto bits=finite_bits.Get<uint8_t>();
    // On this CANN vector primitive, float comparisons consume full 64-lane
    // repeats. Define the rounded tail locally without reading beyond x.
    const uint32_t compared=(count+63)/64*64;
    AscendC::Duplicate(magnitude,0.f,compared);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Abs(magnitude,x,count);AscendC::PipeBarrier<PIPE_V>();
    AscendC::Compares(bits,magnitude,3.40282346638528859812e+38F,AscendC::CMPMODE::LE,compared);
    AscendC::PipeBarrier<PIPE_ALL>();
    auto words=bits.ReinterpretCast<uint64_t>();const uint32_t whole=count/64,tail=count%64;
    for(uint32_t i=0;i<whole;++i)if(words.GetValue(i)!=~uint64_t(0))return false;
    if(tail){const uint64_t mask=(uint64_t(1)<<tail)-1;if((words.GetValue(whole)&mask)!=mask)return false;}
    return true;
  }
 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN,1> input;
  AscendC::TQue<AscendC::QuePosition::VECOUT,1> output;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> storage,finite_bits;
};
} // namespace tide_device
