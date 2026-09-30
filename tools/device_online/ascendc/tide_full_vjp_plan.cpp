#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_full_vjp_plan(GM_ADDR metadata,GM_ADDR count,GM_ADDR kinds,
    GM_ADDR connected,GM_ADDR content_connected,GM_ADDR comparison_connected,GM_ADDR parameter_connected,
    GM_ADDR cursor,GM_ADDR sources,GM_ADDR parameters,GM_ADDR destinations,GM_ADDR owners,GM_ADDR owner_count,
    GM_ADDR branch,GM_ADDR chunks,GM_ADDR error,int64_t capacity,int64_t nodes,int64_t samples,int64_t chunk,int64_t has_tanh) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto m=(__gm__ I*)metadata,k=(__gm__ I*)kinds,pos=(__gm__ I*)cursor;
  auto source=(__gm__ I*)sources,param=(__gm__ I*)parameters,dest=(__gm__ I*)destinations;
  auto list=(__gm__ I*)owners,nowners=(__gm__ I*)owner_count;
  auto g=(__gm__ uint8_t*)connected,h=(__gm__ uint8_t*)content_connected,s=(__gm__ uint8_t*)comparison_connected,w=(__gm__ uint8_t*)parameter_connected;
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;go[0]=0;nowners[0]=0;
  const I size=((__gm__ I*)count)[0];
  if(status[0]==0&&(size<0||size>capacity||pos[0]<0||pos[0]>size))status[0]=2;
  // Whole-tape preflight before any numerical chunk; flags have one writer.
  if(pos[0]==0&&status[0]==0) {
    for(I n=0;n<nodes;++n)if(k[n]<0||k[n]>(has_tanh?1:0)){status[0]=12;break;}
    for(I i=0;i<size&&status[0]==0;++i) {
      const I b=m[i*13],n=m[i*13+1],t=m[i*13+2],active=m[i*13+3];
      if(b<0||b>=samples||n<0||n>=nodes||t<0||(active!=0&&active!=1)||(g[i]&&!active)){status[0]=2;break;}
    }
    if(status[0]==0)for(I i=0;i<size;++i) {
      const I n=m[i*13+1];h[i]=g[i];s[i]=g[i]&&k[n]==1;if(s[i])w[n]=1;
    }
  }
  for(I i=0;i<chunk;++i){source[i]=capacity;param[i]=nodes;dest[i]=capacity+i;list[i]=-1;}
  I used=0;
  while(status[0]==0&&pos[0]<size&&used<chunk) {
    const I i=pos[0]++,n=m[i*13+1];
    if(!g[i]||k[n]!=1)continue;
    source[used]=i;param[used]=n;dest[used]=i;++used;
    bool found=false;for(I j=0;j<nowners[0];++j)found|=list[j]==n;
    if(!found)list[nowners[0]++]=n;
  }
  if(status[0]==0&&used){go[0]=1;++((__gm__ I*)chunks)[0];}
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
