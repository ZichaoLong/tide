#pragma once
#include "kernel_operator.h"
namespace tide_device {
// Contiguous payload tiles; tails never read or overwrite a neighboring row.
class FiberVector {
 public:
  static constexpr uint32_t tile=256;
  __aicore__ inline void init(){pipe.InitBuffer(input,1,tile*4);pipe.InitBuffer(output,1,tile*4);
    pipe.InitBuffer(first,tile*4);pipe.InitBuffer(second,tile*4);}
  __aicore__ inline AscendC::LocalTensor<float> x(){return first.Get<float>();}
  __aicore__ inline AscendC::LocalTensor<float> y(){return second.Get<float>();}
  __aicore__ inline void load(AscendC::LocalTensor<float> to,__gm__ float* from,int64_t offset,uint32_t count){
    AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(from);
    auto v=input.AllocTensor<float>();
    AscendC::DataCopyPad(v,gm[offset],AscendC::DataCopyExtParams{1,count*4,0,0,0},
      AscendC::DataCopyPadExtParams<float>{true,0,uint8_t((8-count%8)%8),0});
    input.EnQue(v);v=input.DeQue<float>();AscendC::Muls(to,v,1.f,count);AscendC::PipeBarrier<PIPE_V>();input.FreeTensor(v);
  }
  __aicore__ inline void save(AscendC::LocalTensor<float> from,__gm__ float* to,int64_t offset,uint32_t count){
    AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(to);auto v=output.AllocTensor<float>();
    AscendC::Muls(v,from,1.f,count);output.EnQue(v);v=output.DeQue<float>();
    AscendC::DataCopyPad(gm[offset],v,AscendC::DataCopyExtParams{1,count*4,0,0,0});output.FreeTensor(v);
  }
  __aicore__ inline void load(AscendC::LocalTensor<float> to,__gm__ half* from,int64_t offset,uint32_t count){
    AscendC::GlobalTensor<half> gm;gm.SetGlobalBuffer(from);auto v=input.AllocTensor<half>();
    AscendC::DataCopyPad(v,gm[offset],AscendC::DataCopyExtParams{1,count*2,0,0,0},
      AscendC::DataCopyPadExtParams<half>{true,0,uint8_t((16-count%16)%16),half(0)});
    input.EnQue(v);v=input.DeQue<half>();AscendC::Cast(to,v,AscendC::RoundMode::CAST_NONE,count);
    AscendC::PipeBarrier<PIPE_V>();input.FreeTensor(v);
  }
  __aicore__ inline void save(AscendC::LocalTensor<float> from,__gm__ half* to,int64_t offset,uint32_t count){
    AscendC::GlobalTensor<half> gm;gm.SetGlobalBuffer(to);auto v=output.AllocTensor<half>();
    AscendC::Cast(v,from,AscendC::RoundMode::CAST_RINT,count);output.EnQue(v);v=output.DeQue<half>();
    AscendC::DataCopyPad(gm[offset],v,AscendC::DataCopyExtParams{1,count*2,0,0,0});output.FreeTensor(v);
  }
 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN,1> input;
  AscendC::TQue<AscendC::QuePosition::VECOUT,1> output;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> first,second;
};
}
