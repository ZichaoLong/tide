#include "kernel_operator.h"
namespace {using I=int64_t;}
// events: [ready row, cache owner, parameter owner, old K, new K, dropped old K,0].
extern "C" __global__ __aicore__ void tide_event_plan(GM_ADDR fibers,GM_ADDR lengths,GM_ADDR mapping,
    GM_ADDR cache_lengths,GM_ADDR windows,GM_ADDR events,GM_ADDR counts,GM_ADDR error,
    int64_t parameters,int64_t capacity,int64_t rows) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto f=(__gm__ I*)fibers,e=(__gm__ I*)events,c=(__gm__ I*)counts;
  auto status=(__gm__ int32_t*)error;c[0]=c[1]=0;
  for(I row=0;row<((__gm__ I*)lengths)[1]&&status[0]==0;++row) {
    const I parameter=((__gm__ I*)mapping)[f[row*4+1]];if(parameter<0)continue;
    const I owner=f[row*4]*parameters+parameter,old=((__gm__ I*)cache_lengths)[owner],window=((__gm__ I*)windows)[parameter];
    if(old<0||old>capacity||c[1]>=rows){status[0]=11;break;}
    for(I j=0;j<c[1];++j)if(e[j*7+1]==owner)status[0]=2;
    I keep=old;if(window>0&&keep>=window)keep=window-1;
    if(keep>=capacity){status[0]=11;break;}
    auto out=e+c[1]*7;out[0]=row;out[1]=owner;out[2]=parameter;out[3]=old;out[4]=keep+1;out[5]=old-keep;out[6]=0;
    ++c[1];c[0]=c[1];
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
