#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_attention_reverse_plan(GM_ADDR lengths,GM_ADDR connections,
    GM_ADDR cursor,GM_ADDR valid,GM_ADDR branch,GM_ADDR connected,GM_ADDR tiles,GM_ADDR error,
    int64_t queries,int64_t capacity,int64_t tile,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)lengths);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;
  auto n=(__gm__ I*)lengths,pos=(__gm__ I*)cursor,v=(__gm__ I*)valid;
  auto on=(__gm__ uint8_t*)connections,out=(__gm__ uint8_t*)connected;
  go[0]=0;if(status[0])return;
  if(mode==0) {
    pos[0]=pos[1]=0;((__gm__ I*)tiles)[0]=0;
    for(I q=0;q<queries;++q){out[q]=on[q];v[q]=0;
      if(n[q]<0||n[q]>capacity||(on[q]&&n[q]==0)){status[0]=2;break;}}
  }else if(mode==1) {
    const I first=pos[0];pos[1]=first;
    for(I q=0;q<queries;++q){const I left=on[q]?n[q]-first:0;v[q]=left<=0?0:left<tile?left:tile;go[0]|=v[q]>0;}
    if(go[0]){pos[0]+=tile;++((__gm__ I*)tiles)[0];}
  }else {pos[0]=pos[1]=0;}
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
