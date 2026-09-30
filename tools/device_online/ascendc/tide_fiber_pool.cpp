#include "kernel_operator.h"
namespace {
using I=int64_t;
__aicore__ inline I slot(__gm__ I* tokens,__gm__ I* atoms,__gm__ I* sources,I token,I inputs) {
  const I row=tokens[token*4],source=atoms[row*6+4]+(atoms[row*6+3]?inputs:0);
  return sources[source*2+1];
}
}
extern "C" __global__ __aicore__ void tide_fiber_pool(GM_ADDR events,GM_ADDR tokens,GM_ADDR counts,
    GM_ADDR atoms,GM_ADDR sources,GM_ADDR kinds,GM_ADDR lengths,GM_ADDR weights,GM_ADDR ids,
    GM_ADDR logits,GM_ADDR probabilities,GM_ADDR coefficients,GM_ADDR error,
    int64_t rows,int64_t slots,int64_t chunk,int64_t inputs,int64_t mode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;if(status[0])return;
  auto e=(__gm__ I*)events,t=(__gm__ I*)tokens,c=(__gm__ I*)counts,a=(__gm__ I*)atoms,s=(__gm__ I*)sources;
  auto k=(__gm__ I*)kinds,n=(__gm__ I*)lengths,index=(__gm__ I*)ids;
  auto w=(__gm__ float*)weights,l=(__gm__ float*)logits,p=(__gm__ float*)probabilities,out=(__gm__ float*)coefficients;
  const float minus_inf=AscendC::GetScalarBitcodeValue<uint32_t,float>(0xff800000);
  if(mode==0)for(I row=0;row<c[0];++row) {
    const I owner=t[row*4+3],local=slot(t,a,s,row,inputs);
    if(local<0||local>=n[owner]){status[0]=2;break;}
    out[row]=k[owner]==2?w[owner*slots+local]:1.f;
  }
  else for(I row=0;row<chunk;++row) {
    const I event=index[row];
    if(mode==1)for(I j=0;j<slots;++j)l[row*slots+j]=minus_inf;
    if(event<0){if(mode==1)l[row*slots]=0.f;continue;}
    const I owner=e[event*7+2],start=e[event*7+5],count=e[event*7+4]-e[event*7+3];
    if(mode==1&&k[owner]==4)for(I j=0;j<n[owner];++j)l[row*slots+j]=w[owner*slots+j];
    for(I i=start;i<start+count;++i) {
      const I local=slot(t,a,s,i,inputs);
      if(mode==1&&k[owner]==3)l[row*slots+local]=w[owner*slots+local];
      if(mode==2)out[i]=p[row*slots+local];
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
