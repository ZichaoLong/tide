#include "kernel_operator.h"
namespace {
using I=int64_t;
class OnlineSoftmax {
 public:
  __aicore__ inline void run(GM_ADDR scores,GM_ADDR valid,GM_ADDR normalization,GM_ADDR weights,I rows,I heads,I tile) {
    pipe.InitBuffer(input,1,256*4);pipe.InitBuffer(output,1,256*4);
    pipe.InitBuffer(first,256*4);pipe.InitBuffer(second,256*4);
    pipe.InitBuffer(temporary,256*4);pipe.InitBuffer(summary,32);pipe.InitBuffer(stat_buffer,32);
    auto x=first.Get<float>(),y=second.Get<float>(),tmp=temporary.Get<float>(),s=summary.Get<float>(),stat=stat_buffer.Get<float>();
    for(I row=AscendC::GetBlockIdx();row<rows*heads;row+=AscendC::GetBlockNum()) {
      const I size=((__gm__ I*)valid)[row/heads];
      load(stat,(__gm__ float*)normalization,row*8,8);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      const float old_max=stat.GetValue(0),old_sum=stat.GetValue(1);float scale=1.f,new_max=old_max,new_sum=old_sum;
      AscendC::Duplicate(y,0.f,uint32_t(tile));AscendC::PipeBarrier<PIPE_V>();
      if(size>0) {
        stat.SetValue(3,1.f); // Distinguish real zero-mass work from query padding.
        load(x,(__gm__ float*)scores,row*tile,uint32_t(size));
        AscendC::ReduceMax(s,x,tmp,int32_t(size));AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        const float maximum=s.GetValue(0);
        const float minus_inf=AscendC::GetScalarBitcodeValue<uint32_t,float>(0xff800000);
        // Finite repeated bias decay can saturate old logits to -infinity.
        // An all-minus-inf tile contributes no mass; a later tile may still
        // contain finite current keys. Never evaluate -inf - -inf here.
        if(maximum!=minus_inf) {
        new_max=old_sum>0&&old_max>maximum?old_max:maximum;
        AscendC::Adds(x,x,-new_max,uint32_t(size));AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(y,x,uint32_t(size));AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(s,y,tmp,int32_t(size));AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        const float tile_sum=s.GetValue(0);scale=0.f;
        if(old_sum>0) {
          s.SetValue(0,old_max-new_max);
          AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
          AscendC::Exp(s,s,1);AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
          scale=s.GetValue(0);
        }
        new_sum=old_sum*scale+tile_sum;
        }
      }
      stat.SetValue(0,new_max);stat.SetValue(1,new_sum);stat.SetValue(2,scale);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      save(y,(__gm__ float*)weights,row*tile,uint32_t(tile));save(stat,(__gm__ float*)normalization,row*8,8);
    }
  }
 private:
  __aicore__ inline void load(AscendC::LocalTensor<float> to,__gm__ float* from,I offset,uint32_t count) {
    AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(from);auto v=input.AllocTensor<float>();
    AscendC::DataCopyPad(v,gm[offset],AscendC::DataCopyExtParams{1,count*4,0,0,0},
      AscendC::DataCopyPadExtParams<float>{true,0,uint8_t((8-count%8)%8),0});
    input.EnQue(v);v=input.DeQue<float>();AscendC::Muls(to,v,1.f,count);AscendC::PipeBarrier<PIPE_V>();input.FreeTensor(v);
  }
  __aicore__ inline void save(AscendC::LocalTensor<float> from,__gm__ float* to,I offset,uint32_t count) {
    AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(to);auto v=output.AllocTensor<float>();
    AscendC::Muls(v,from,1.f,count);output.EnQue(v);v=output.DeQue<float>();
    AscendC::DataCopyPad(gm[offset],v,AscendC::DataCopyExtParams{1,count*4,0,0,0});output.FreeTensor(v);
  }
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN,1> input;
  AscendC::TQue<AscendC::QuePosition::VECOUT,1> output;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> first,second,temporary,summary,stat_buffer;
};
}
extern "C" __global__ __aicore__ void tide_attention_softmax(GM_ADDR scores,GM_ADDR valid,
    GM_ADDR normalization,GM_ADDR weights,GM_ADDR error,int64_t rows,int64_t heads,int64_t tile) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)valid);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  OnlineSoftmax op;op.run(scores,valid,normalization,weights,rows,heads,tile);
}
