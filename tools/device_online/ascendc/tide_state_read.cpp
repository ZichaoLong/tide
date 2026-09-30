#include "kernel_operator.h"
namespace {
using I=int64_t;
__aicore__ inline bool finite(float f){return (AscendC::GetScalarBitcodeValue<float,uint32_t>(f)&0x7f800000)!=0x7f800000;}
}
// Fibers are ordered (sample,node,time). Multi-time old/proposal Read is legal
// only for observe-all + no clear; DeviceReady narrows all other such regions
// to one complete frame before this task. Scratch never changes persistent state.
extern "C" __global__ __aicore__ void tide_state_read(GM_ADDR fibers,GM_ADDR lengths,
    GM_ADDR content,GM_ADDR reads,GM_ADDR modes,GM_ADDR config,GM_ADDR coefficients,
    GM_ADDR scratch,GM_ADDR scores,GM_ADDR error,int64_t capacity,int64_t width,int64_t nodes,int64_t samples) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)fibers);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;
  auto f=(__gm__ I*)fibers,m=(__gm__ I*)modes,settings=(__gm__ I*)config;
  auto h=(__gm__ float*)content,w=(__gm__ float*)reads,a=(__gm__ float*)coefficients;
  auto state=(__gm__ float*)scratch,d=(__gm__ float*)scores;
  I count=((__gm__ I*)lengths)[1];
  if(status[0]==0&&(count<0||count>capacity))status[0]=2;
  for(I i=0;i<count&&status[0]==0;++i) {
    I b=f[i*4],n=f[i*4+1];
    if(b<0||b>=samples||n<0||n>=nodes){status[0]=2;break;}
    I mode=m[n],key=b*nodes+n;float score=0;
    if(mode>=0)for(I j=0;j<width;++j) {
      float value=h[i*width+j];
      if(mode) {
        float old=state[key*width+j],proposal=old;
        if(settings[n*3]){float decayed=a[n*width+j]*old;proposal=decayed+value;}
        value=mode==1?old:proposal;state[key*width+j]=proposal;
      }
      float product=value*w[n*width+j];score=score+product;
    }
    d[i]=score;if(!finite(score))status[0]=6;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
