#pragma once
#include "kernel_operator.h"

namespace tide_device {
// State arithmetic uses FP32 vector instructions. Half payloads round at each
// declared product/add, including every Add-repeat tick and every event. This
// keeps rounding independent of how many times share one physical batch.
class StatePayload {
 public:
  static constexpr uint32_t tile=256;
  __aicore__ inline void init(AscendC::TPipe& pipe) {
    pipe.InitBuffer(input,1,tile*4);pipe.InitBuffer(output,1,tile*4);
    pipe.InitBuffer(rounded,tile*2);
  }
  template<class T>
  __aicore__ inline void load(AscendC::LocalTensor<float> to,__gm__ T* from,int64_t offset,uint32_t count) {
    AscendC::GlobalTensor<T> gm;gm.SetGlobalBuffer(from);auto v=input.AllocTensor<T>();
    constexpr uint32_t block=32/sizeof(T);
    AscendC::DataCopyPad(v,gm[offset],AscendC::DataCopyExtParams{1,count*uint32_t(sizeof(T)),0,0,0},
      AscendC::DataCopyPadExtParams<T>{true,0,uint8_t((block-count%block)%block),T(0)});
    input.EnQue(v);v=input.DeQue<T>();
    if constexpr(sizeof(T)==2)AscendC::Cast(to,v,AscendC::RoundMode::CAST_NONE,count);
    else AscendC::Muls(to,v,1.f,count);
    AscendC::PipeBarrier<PIPE_V>();input.FreeTensor(v);
  }
  template<class T>
  __aicore__ inline void save(AscendC::LocalTensor<float> from,__gm__ T* to,int64_t offset,uint32_t count) {
    AscendC::GlobalTensor<T> gm;gm.SetGlobalBuffer(to);auto v=output.AllocTensor<T>();
    if constexpr(sizeof(T)==2)AscendC::Cast(v,from,AscendC::RoundMode::CAST_RINT,count);
    else AscendC::Muls(v,from,1.f,count);
    output.EnQue(v);v=output.DeQue<T>();
    AscendC::DataCopyPad(gm[offset],v,AscendC::DataCopyExtParams{1,count*uint32_t(sizeof(T)),0,0,0});output.FreeTensor(v);
  }
  template<class T>
  __aicore__ inline void round(AscendC::LocalTensor<float> value,uint32_t count) {
    if constexpr(sizeof(T)==2) {
      auto tmp=rounded.Get<half>();AscendC::Cast(tmp,value,AscendC::RoundMode::CAST_RINT,count);
      AscendC::PipeBarrier<PIPE_V>();AscendC::Cast(value,tmp,AscendC::RoundMode::CAST_NONE,count);
      AscendC::PipeBarrier<PIPE_V>();
    }
  }
 private:
  AscendC::TQue<AscendC::QuePosition::VECIN,1> input;
  AscendC::TQue<AscendC::QuePosition::VECOUT,1> output;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> rounded;
};
} // namespace tide_device
