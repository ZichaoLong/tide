#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_fiber_chunk(GM_ADDR events,GM_ADDR tokens,GM_ADDR counts,
    GM_ADDR heads,GM_ADDR cursor,GM_ADDR source,GM_ADDR parameter,GM_ADDR destination,GM_ADDR ids,
    GM_ADDR branch,GM_ADDR chunks,GM_ADDR error,int64_t rows,int64_t parameters,int64_t chunk,int64_t mode,int64_t target_heads) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)cursor);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;
  auto e=(__gm__ I*)events,t=(__gm__ I*)tokens,c=(__gm__ I*)counts,pos=(__gm__ I*)cursor;
  auto s=(__gm__ I*)source,p=(__gm__ I*)parameter,d=(__gm__ I*)destination,index=(__gm__ I*)ids;
  go[0]=0;for(I i=0;i<chunk;++i){s[i]=rows;p[i]=parameters;d[i]=rows+i;index[i]=-1;}
  I filled=0;const I length=c[mode>=2?1:0];
  while(status[0]==0&&pos[0]<length&&filled<chunk) {
    const I row=pos[0]++,owner=mode>=2?e[row*7+2]:t[row*4+3];
    if((mode==1||mode==3)&&((__gm__ I*)heads)[owner]!=target_heads)continue;
    index[filled]=row;p[filled]=owner;
    s[filled]=mode>=2?e[row*7]:mode==0?t[row*4]:row;
    d[filled]=mode>=2?e[row*7]:row;++filled;
  }
  if(status[0]==0&&filled) {
    auto n=(__gm__ I*)chunks;if(n[0]==I(0x7fffffffffffffff))status[0]=5;
    else {++n[0];go[0]=1;}
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
