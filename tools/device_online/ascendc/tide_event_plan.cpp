#include "kernel_operator.h"
namespace {using I=int64_t;}
// Compact node-time sequences; see event_sequence.h for the seven fields.
extern "C" __global__ __aicore__ void tide_event_plan(GM_ADDR fibers,GM_ADDR lengths,GM_ADDR mapping,
    GM_ADDR cache_lengths,GM_ADDR windows,GM_ADDR config,GM_ADDR events,GM_ADDR counts,GM_ADDR error,
    int64_t parameters,int64_t capacity,int64_t rows) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto f=(__gm__ I*)fibers,e=(__gm__ I*)events,c=(__gm__ I*)counts;
  auto status=(__gm__ int32_t*)error;c[0]=c[1]=0;
  for(I row=0;row<((__gm__ I*)lengths)[1]&&status[0]==0;++row) {
    const I parameter=((__gm__ I*)mapping)[f[row*4+1]];if(parameter<0)continue;
    const I owner=f[row*4]*parameters+parameter,initial=((__gm__ I*)cache_lengths)[owner],window=((__gm__ I*)windows)[parameter];
    if(initial<0||initial>capacity||c[1]>=rows){status[0]=11;break;}
    const bool next=c[1]>0&&e[(c[1]-1)*7+1]==owner;
    const I first=next?e[(c[1]-1)*7+6]:c[1],ordinal=c[1]-first;
    auto cfg=(__gm__ I*)config;
    if(next&&(!cfg[parameter*2]||cfg[parameter*2+1])){status[0]=2;break;}
    if(ordinal>=I(0x7fffffffffffffff)-initial){status[0]=5;break;}
    const I old=next?e[(c[1]-1)*7+4]:initial;
    I keep=old;if(window>0&&keep>=window)keep=window-1;
    if(keep>=capacity){status[0]=11;break;}
    auto out=e+c[1]*7;out[0]=row;out[1]=owner;out[2]=parameter;out[3]=old;out[4]=keep+1;out[5]=initial+ordinal;out[6]=first;
    ++c[1];c[0]=c[1];
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
