#include "kernel_operator.h"
namespace {
using I=int64_t;
class ScaleVectors {
 public:
  __aicore__ inline void init(){pipe.InitBuffer(input,1,256*4);pipe.InitBuffer(output,1,256*4);pipe.InitBuffer(storage,3*256*4);}
  __aicore__ inline AscendC::LocalTensor<float> at(I index){return storage.Get<float>()[index*256];}
  __aicore__ inline void load(AscendC::LocalTensor<float> to,__gm__ float* from,I offset,uint32_t count){
    AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(from);auto x=input.AllocTensor<float>();
    AscendC::DataCopyPad(x,gm[offset],AscendC::DataCopyExtParams{1,count*4,0,0,0},
      AscendC::DataCopyPadExtParams<float>{true,0,uint8_t((8-count%8)%8),0});
    input.EnQue(x);x=input.DeQue<float>();AscendC::Muls(to,x,1.f,count);AscendC::PipeBarrier<PIPE_V>();input.FreeTensor(x);
  }
  __aicore__ inline void save(AscendC::LocalTensor<float> from,__gm__ float* to,I offset,uint32_t count){
    AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(to);auto x=output.AllocTensor<float>();AscendC::Muls(x,from,1.f,count);
    output.EnQue(x);x=output.DeQue<float>();AscendC::DataCopyPad(gm[offset],x,AscendC::DataCopyExtParams{1,count*4,0,0,0});output.FreeTensor(x);
  }
 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN,1> input;
  AscendC::TQue<AscendC::QuePosition::VECOUT,1> output;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> storage;
};
}
extern "C" __global__ __aicore__ void tide_graph_reverse_scales(GM_ADDR messages,GM_ADDR heads,GM_ADDR next,
    GM_ADDR gradients,GM_ADDR connected,GM_ADDR aggregate_partials,GM_ADDR full_values,GM_ADDR partials,GM_ADDR error,
    int64_t parameters,int64_t width) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)messages);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto h=(__gm__ I*)heads,n=(__gm__ I*)next,m=(__gm__ I*)messages;
  const I tiles=(width+255)/256;
  ScaleVectors vector;vector.init();auto total=vector.at(0),part=vector.at(1),value=vector.at(2);
  for(I task=AscendC::GetBlockIdx();task<parameters*tiles;task+=AscendC::GetBlockNum()) {
    const I param=task/tiles,start=(task%tiles)*256;const uint32_t size=width-start<256?width-start:256;
    AscendC::Duplicate(total,0.f,size);AscendC::PipeBarrier<PIPE_V>();
    for(I entry=h[param];entry>=0;entry=n[entry]) {
      const I row=entry/2;if(!((__gm__ uint8_t*)connected)[row])continue;
      if(entry%2==0)vector.load(part,(__gm__ float*)aggregate_partials,row*width+start,size);
      else {vector.load(part,(__gm__ float*)gradients,row*width+start,size);
        vector.load(value,(__gm__ float*)full_values,m[row*4+1]*width+start,size);
        AscendC::Mul(part,part,value,size);AscendC::PipeBarrier<PIPE_V>();}
      AscendC::Add(total,total,part,size);AscendC::PipeBarrier<PIPE_V>();
    }
    vector.save(total,(__gm__ float*)partials,param*width+start,size);
  }
}
