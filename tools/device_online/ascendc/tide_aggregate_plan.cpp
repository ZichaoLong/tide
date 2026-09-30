#include "kernel_operator.h"
namespace {using I=int64_t;}
// mode0 mean coefficients; mode1 actual event chunk and complete coefficient
// domain; mode2 scatter probabilities to physical atom rows. Sum stays unchanged.
extern "C" __global__ __aicore__ void tide_aggregate_plan(GM_ADDR fibers,GM_ADDR offsets,GM_ADDR counts,
    GM_ADDR keys,GM_ADDR sources,GM_ADDR kinds,GM_ADDR lengths,GM_ADDR weights,GM_ADDR ids,
    GM_ADDR cursor,GM_ADDR branch,GM_ADDR chunks,GM_ADDR logits,GM_ADDR probabilities,GM_ADDR totals,
    GM_ADDR coefficients,GM_ADDR error,int64_t slots,int64_t chunk,int64_t mode,int64_t target) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)fibers);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;
  auto f=(__gm__ I*)fibers,o=(__gm__ I*)offsets,c=(__gm__ I*)counts,s=(__gm__ I*)sources,k=(__gm__ I*)keys;
  auto type=(__gm__ I*)kinds,n=(__gm__ I*)lengths,index=(__gm__ I*)ids,pos=(__gm__ I*)cursor;
  auto w=(__gm__ float*)weights,l=(__gm__ float*)logits,p=(__gm__ float*)probabilities,coe=(__gm__ float*)coefficients;
  if(mode==1)go[0]=0;
  if(status[0])return;
  if(mode==0)for(I event=0;event<c[1];++event) {
    const I node=f[event*4+1],size=o[event+1]-o[event];
    for(I a=o[event];a<o[event+1];++a) {
      const I slot=s[k[a]*2+1];if(slot<0||slot>=n[node]){status[0]=2;break;}
      coe[a]=type[node]==1?1.f/static_cast<float>(size):1.f;
    }
  }
  if(mode==1) {
    I found=0;
    while(pos[0]<c[1]&&found<chunk) {
      const I event=pos[0]++;if(type[f[event*4+1]]==target)index[found++]=event;
    }
    for(I row=found;row<chunk;++row)index[row]=-1;
    if(found){go[0]=1;++((__gm__ I*)chunks)[0];}
  }
  if(mode==1||mode==2)for(I row=0;row<chunk;++row) {
    const I event=index[row];
    if(mode==1)for(I j=0;j<slots;++j)l[row*slots+j]=AscendC::GetScalarBitcodeValue<uint32_t,float>(0xff800000);
    if(event<0){if(mode==1)l[row*slots]=0.f;continue;}
    const I node=f[event*4+1];
    if(mode==1&&target==4)for(I j=0;j<n[node];++j)l[row*slots+j]=w[node*slots+j];
    if(mode==2&&target==2&&!(((__gm__ float*)totals)[row]>0.f)){status[0]=13;break;}
    for(I a=o[event];a<o[event+1];++a) {
      const I slot=s[k[a]*2+1];
      if(mode==1&&target!=4)l[row*slots+slot]=w[node*slots+slot];
      if(mode==2)coe[a]=p[row*slots+slot];
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
