#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_fiber_vjp_plan(GM_ADDR counts,GM_ADDR slots,GM_ADDR lengths,
    GM_ADDR old_lengths,GM_ADDR ticks,GM_ADDR pool_kinds,GM_ADDR pool_lengths,GM_ADDR proposal_on,
    GM_ADDR key_on,GM_ADDR value_on,GM_ADDR bias_on,GM_ADDR row_on,GM_ADDR cache_on,GM_ADDR parameter_on,
    GM_ADDR cursor,GM_ADDR plan,GM_ADDR branch,GM_ADDR chunks,GM_ADDR error,
    int64_t batch,int64_t sources,int64_t capacity,int64_t domain,int64_t chunk,int64_t max_ticks,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)counts);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;
  auto c=(__gm__ I*)counts,s=(__gm__ I*)slots,l=(__gm__ I*)lengths,o=(__gm__ I*)old_lengths;
  auto t=(__gm__ I*)ticks,k=(__gm__ I*)pool_kinds,n=(__gm__ I*)pool_lengths,at=(__gm__ I*)cursor;
  auto p=(__gm__ uint8_t*)proposal_on,ko=(__gm__ uint8_t*)key_on,vo=(__gm__ uint8_t*)value_on,bo=(__gm__ uint8_t*)bias_on;
  auto input=(__gm__ uint8_t*)row_on,co=(__gm__ uint8_t*)cache_on,po=(__gm__ uint8_t*)parameter_on;
  go[0]=0;
  if(mode==0) {
    at[0]=0;((__gm__ I*)chunks)[0]=0;
    for(I b=0;b<batch&&!status[0];++b) {
      if(c[b]<0||c[b]>sources||o[b]<0||l[b]!=o[b]+c[b]||l[b]>capacity||t[b]<0||t[b]>max_ticks
          ||k[b]<0||k[b]>4||n[b]<0||n[b]>domain||(c[b]==0&&p[b])){status[0]=2;break;}
      I previous=-1;
      for(I j=0;j<c[b];++j){const I slot=s[b*sources+j];
        if(slot<=previous||slot>=n[b]){status[0]=2;break;}previous=slot;}
      input[b]=p[b]||ko[b]||vo[b];
      co[b*3]=p[b]||ko[b];co[b*3+1]=p[b]||vo[b];co[b*3+2]=p[b]||bo[b];
      po[b*6]=po[b*6+1]=input[b]&&c[b]>0;
      po[b*6+2]=po[b*6+3]=p[b];po[b*6+4]=co[b*3+2]&&o[b]>0&&t[b]>0;
      po[b*6+5]=p[b]&&k[b]>=2;
    }
  }else if(mode==3)at[0]=0;
  else {
    auto rows=(__gm__ I*)plan;for(I j=0;j<chunk*3;++j)rows[j]=-1;
    I count=0;
    while(at[0]<batch*sources&&count<chunk&&!status[0]) {
      const I x=at[0]++,b=x/sources,j=x%sources;
      if(j>=c[b]||!(mode==1?p[b]:input[b]))continue;
      rows[count*3]=b;rows[count*3+1]=j;rows[count*3+2]=s[b*sources+j];++count;
    }
    if(count&&!status[0]){go[0]=1;++((__gm__ I*)chunks)[0];}
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
