#include "kernel_operator.h"
#include "state_clock.h"
namespace {
using I=int64_t;
__aicore__ inline bool finite(float f){return (AscendC::GetScalarBitcodeValue<float,uint32_t>(f)&0x7f800000)!=0x7f800000;}
}
// Fibers are ordered (sample,node,time). Multi-time old/proposal Read is legal
// only for observe-all + no clear; DeviceReady narrows all other such regions
// to one complete frame before this task. Scratch never changes persistent state.
extern "C" __global__ __aicore__ void tide_state_read(GM_ADDR fibers,GM_ADDR lengths,
    GM_ADDR content,GM_ADDR reads,GM_ADDR modes,GM_ADDR kinds,GM_ADDR config,GM_ADDR coefficients,GM_ADDR retention,GM_ADDR policy,
    GM_ADDR scratch,GM_ADDR clocks,GM_ADDR scores,GM_ADDR steps,GM_ADDR proposals,GM_ADDR error,int64_t capacity,int64_t width,int64_t nodes,int64_t samples,int64_t max_ticks,int64_t vectorized) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::TPipe pipe;AscendC::TBuf<AscendC::QuePosition::VECCALC> root_buffer;
  pipe.InitBuffer(root_buffer,32);auto root=root_buffer.Get<float>();
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)fibers);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;
  auto f=(__gm__ I*)fibers,m=(__gm__ I*)modes,settings=(__gm__ I*)config;
  auto h=(__gm__ float*)content,w=(__gm__ float*)reads,a=(__gm__ float*)coefficients;
  auto state=(__gm__ float*)scratch,d=(__gm__ float*)scores;auto t=(__gm__ I*)clocks;
  I count=((__gm__ I*)lengths)[1];
  if(status[0]==0&&(count<0||count>capacity))status[0]=2;
  for(I i=0;i<count&&status[0]==0;++i) {
    I b=f[i*4],n=f[i*4+1],time=f[i*4+2];
    if(b<0||b>=samples||n<0||n>=nodes){status[0]=2;break;}
    I mode=m[n],key=b*nodes+n,kind=settings[n*3];float score=0;
    const bool norm=((__gm__ I*)kinds)[n];
    uint64_t ticks=0;
    if(mode>0) {
      if(t[key*2]< -1||t[key*2]>=time||t[key*2+1]<0){status[0]=2;break;}
      if(kind&&t[key*2+1]==I(0x7fffffffffffffff)){status[0]=5;break;}
      I local,local_old;
      if(!tide_device::local_time(time,(__gm__ I*)policy,n,local)
          ||!tide_device::local_time(t[key*2],(__gm__ I*)policy,n,local_old)){status[0]=9;break;}
      ticks=(uint64_t(local)+1)-uint64_t(local_old+1);
      if(kind==2&&ticks>uint64_t(max_ticks)){status[0]=8;break;}
    }
    if(vectorized)((__gm__ I*)steps)[i]=kind==2?I(ticks):0;
    if(!vectorized&&mode>=0)for(I j=0;j<width;++j) {
      float value=h[i*width+j];
      if(mode) {
        float old=state[key*width+j],proposal=old;
        if(kind==1){float decayed=a[n*width+j]*old;proposal=decayed+value;}
        if(kind==2){float rho=((__gm__ float*)retention)[n];
          for(uint64_t tick=0;tick<ticks;++tick)proposal=proposal*rho;
          proposal=value+proposal;}
        if(kind==3)proposal=((__gm__ float*)proposals)[i*width+j];
        value=mode==1?old:proposal;state[key*width+j]=proposal;
      }
      float product=norm?value*value:value*w[n*width+j];score=score+product;
    }
    if(!vectorized&&norm&&mode>=0) {
      root.SetValue(0,score);
      AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
      AscendC::Sqrt(root,root,1);
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      score=root.GetValue(0);
    }
    if(!vectorized){d[i]=score;if(!finite(score))status[0]=6;}
    if(mode>0&&kind){t[key*2]=time;++t[key*2+1];}
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
