#include "fiber_vector.h"
extern "C" __global__ __aicore__ void tide_parameter_publish(GM_ADDR plan,GM_ADDR tiles,GM_ADDR values,GM_ADDR weights,GM_ADDR biases,
    GM_ADDR decay,GM_ADDR retention,GM_ADDR read,GM_ADDR sources,GM_ADDR emission,GM_ADDR lh_weights,GM_ADDR lh_biases,GM_ADDR gate,GM_ADDR up,GM_ADDR down,GM_ADDR aggregate,
    GM_ADDR fiber_qkv,GM_ADDR fiber_qkv_bias,GM_ADDR fiber_out,GM_ADDR fiber_out_bias,GM_ADDR fiber_decay,GM_ADDR fiber_pool,GM_ADDR emission_weights,GM_ADDR emission_biases,GM_ADDR error,int64_t count,int64_t tasks,int64_t fp16) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)plan);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  if(((__gm__ int32_t*)error)[0])return;
  auto table=(__gm__ I*)plan,offset=(__gm__ I*)tiles;tide_device::FiberVector vector;vector.init();auto x=vector.x();
  for(I task=AscendC::GetBlockIdx();task<tasks;task+=AscendC::GetBlockNum()) {
    I lo=0,hi=count;while(lo+1<hi){I mid=lo+(hi-lo)/2;if(offset[mid]<=task)lo=mid;else hi=mid;}
    const I i=lo,start=(task-offset[i])*256,remaining=table[i*4+3]-start,bank=table[i*4+1];
    const uint32_t size=remaining<256?remaining:256;
    auto destination=(bank==0?weights:bank==1?biases:bank==2?decay:bank==3?retention:bank==4?read:bank==5?sources:bank==6?emission:bank==7?lh_weights:bank==8?lh_biases:bank==9?gate:bank==10?up:bank==11?down:bank==12?aggregate:
      bank==13?fiber_qkv:bank==14?fiber_qkv_bias:bank==15?fiber_out:bank==16?fiber_out_bias:bank==17?fiber_decay:bank==18?fiber_pool:bank==19?emission_weights:emission_biases);
    vector.load(x,(__gm__ float*)values,table[i*4]+start,size);
    if(fp16&&bank!=12&&bank!=18)vector.save(x,(__gm__ half*)destination,table[i*4+2]+start,size);
    else {
      if(fp16) {
        // FP32 normalization banks contain the widened payload parameter,
        // never the higher-precision master that the public model cannot store.
        auto rounded=vector.y().ReinterpretCast<half>();
        AscendC::Cast(rounded,x,AscendC::RoundMode::CAST_RINT,size);AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(x,rounded,AscendC::RoundMode::CAST_NONE,size);AscendC::PipeBarrier<PIPE_V>();
      }
      vector.save(x,(__gm__ float*)destination,table[i*4+2]+start,size);
    }
  }
}
