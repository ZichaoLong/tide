#include "kernel_operator.h"
// One metadata writer: stable connected rows, compact node IDs and unique
// scratch destinations. Numerical payload is gathered by packed operators.
extern "C" __global__ __aicore__ void tide_full_reverse_pack(GM_ADDR metadata,GM_ADDR count,
    GM_ADDR connected,GM_ADDR mapping,GM_ADDR packed,GM_ADDR packed_count,GM_ADDR packed_on,
    GM_ADDR sources,GM_ADDR destinations,GM_ADDR branch,GM_ADDR work,GM_ADDR error,
    int64_t capacity,int64_t nodes,int64_t local_nodes) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto m=(__gm__ I*)metadata,ids=(__gm__ I*)mapping,out=(__gm__ I*)packed,n=(__gm__ I*)packed_count;
  auto src=(__gm__ I*)sources,dst=(__gm__ I*)destinations;
  auto on=(__gm__ uint8_t*)connected,po=(__gm__ uint8_t*)packed_on;
  auto go=(__gm__ int32_t*)branch,status=(__gm__ int32_t*)error;
  n[0]=0;go[0]=0;
  for(I i=0;i<capacity;++i){src[i]=capacity;dst[i]=capacity+i;po[i]=0;for(I j=0;j<13;++j)out[i*13+j]=0;}
  const I size=((__gm__ I*)count)[0];
  if(!status[0]&&(size<0||size>capacity))status[0]=2;
  for(I i=0;i<size&&!status[0];++i) {
    const I node=m[i*13+1];
    if(node<0||node>=nodes||(on[i]&&m[i*13+3]!=1)){status[0]=2;break;}
    const I local=ids[node];if(local<0||!on[i])continue;
    if(local>=local_nodes){status[0]=2;break;}
    const I row=n[0]++;src[row]=i;dst[row]=i;po[row]=1;
    for(I j=0;j<13;++j)out[row*13+j]=m[i*13+j];out[row*13+1]=local;
  }
  if(!status[0]&&n[0]){go[0]=1;((__gm__ I*)work)[0]+=n[0];++((__gm__ I*)work)[1];}
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
