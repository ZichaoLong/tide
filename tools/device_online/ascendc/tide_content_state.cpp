#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_content_state(GM_ADDR fibers,GM_ADDR lengths,
    GM_ADDR content,GM_ADDR scores,GM_ADDR controls,GM_ADDR active,GM_ADDR config,GM_ADDR coefficients,
    GM_ADDR state,GM_ADDR clocks,GM_ADDR present,GM_ADDR action_coordinates,GM_ADDR comparisons,
    GM_ADDR event_meta,GM_ADDR event_values,GM_ADDR stage,GM_ADDR error,int64_t capacity,int64_t width,int64_t nodes,int64_t samples) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)clocks);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto fs=(__gm__ I*)fibers,cfg=(__gm__ I*)config,t=(__gm__ I*)clocks;
  auto s=(__gm__ float*)state,h=(__gm__ float*)content,a=(__gm__ float*)coefficients;
  auto valid=(__gm__ uint8_t*)present,on=(__gm__ uint8_t*)active;
  auto events=(__gm__ I*)event_meta,actions=(__gm__ I*)action_coordinates;auto v=(__gm__ float*)event_values;
  I count=((__gm__ I*)lengths)[1],stride=5*width+2;
  for(I i=0;i<count&&status[0]==0;++i) {
    I b=fs[i*4],n=fs[i*4+1],time=fs[i*4+2];
    if(b<0||b>=samples||n<0||n>=nodes||time<0){status[0]=2;break;}
    I key=b*nodes+n,old_time=t[key*2],old_count=t[key*2+1];
    bool ema=cfg[n*3],clear=cfg[n*3+1]&&on[i],adopt=cfg[n*3+2]||on[i];
    if(old_count<0||old_time>=time){status[0]=2;break;}
    if(ema&&old_count==I(0x7fffffffffffffff)){status[0]=5;break;}
    I proposed_time=ema?time:old_time,proposed_count=old_count+(ema?1:0);
    I next_time=adopt?proposed_time:old_time,next_count=adopt?proposed_count:old_count;
    auto row=events+i*13;row[12]=((__gm__ I*)stage)[0];
    row[0]=b;row[1]=n;row[2]=time;row[3]=on[i];row[4]=old_time;row[5]=old_count;
    row[6]=proposed_time;row[7]=proposed_count;row[8]=next_time;row[9]=next_count;row[10]=next_time;row[11]=next_count;
    for(I j=0;j<width;++j) {
      float old=s[key*width+j],proposal=old;
      if(ema){float decayed=a[n*width+j]*old;proposal=decayed+h[i*width+j];}
      float comparison=adopt?proposal:old,next=clear?comparison*0.0f:comparison;
      v[i*stride+j]=h[i*width+j];v[i*stride+width+j]=old;v[i*stride+2*width+j]=proposal;
      v[i*stride+3*width+j]=comparison;v[i*stride+4*width+j]=next;s[key*width+j]=next;
      ((__gm__ float*)comparisons)[i*width+j]=comparison;
    }
    v[i*stride+5*width]=((__gm__ float*)scores)[i];v[i*stride+5*width+1]=((__gm__ float*)controls)[i];
    t[key*2]=next_time;t[key*2+1]=next_count;valid[key]=1;
    actions[i*4]=b;actions[i*4+1]=n;actions[i*4+2]=time;actions[i*4+3]=time;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
