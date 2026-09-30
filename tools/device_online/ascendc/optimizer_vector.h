#pragma once
#include "kernel_operator.h"
namespace tide_device {
class OptimizerVector {
 public:
  __aicore__ inline void init(){pipe.InitBuffer(input,1,1024);pipe.InitBuffer(output,1,1024);pipe.InitBuffer(storage,8*1024);}
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
    AscendC::PipeBarrier<PIPE_ALL>();
    for(uint32_t i=0;i<count;++i)if((AscendC::GetScalarBitcodeValue<float,uint32_t>(x.GetValue(i))&0x7f800000)==0x7f800000)return false;
    return true;
  }
 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN,1> input;
  AscendC::TQue<AscendC::QuePosition::VECOUT,1> output;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> storage;
};
} // namespace tide_device
