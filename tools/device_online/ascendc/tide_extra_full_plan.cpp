#include "kernel_operator.h"
extern "C" __global__ __aicore__ void tide_extra_full_plan(GM_ADDR metadata,GM_ADDR count,GM_ADDR kinds,GM_ADDR mapping,
    GM_ADDR connected,GM_ADDR content_connected,GM_ADDR comparison_connected,GM_ADDR parameter_connected,
    GM_ADDR cursor,GM_ADDR sources,GM_ADDR parameters,GM_ADDR destinations,GM_ADDR owners,GM_ADDR owner_count,
    GM_ADDR branch,GM_ADDR chunks,GM_ADDR error,int64_t capacity,int64_t nodes,int64_t chunk,int64_t target,
    int64_t parameter_count,int64_t residual) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto m=(__gm__ I*)metadata,k=(__gm__ I*)kinds,map=(__gm__ I*)mapping,pos=(__gm__ I*)cursor;
  auto s=(__gm__ I*)sources,p=(__gm__ I*)parameters,d=(__gm__ I*)destinations,o=(__gm__ I*)owners,n=(__gm__ I*)owner_count;
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;go[0]=0;n[0]=0;
  const I rows=((__gm__ I*)count)[0];
  for(I i=0;i<chunk;++i){s[i]=capacity;p[i]=parameter_count;d[i]=capacity+i;o[i]=-1;}
  if(status[0])return;
  if(rows<0||rows>capacity||pos[0]<0||pos[0]>rows){status[0]=2;return;}
  I used=0;
  while(pos[0]<rows&&used<chunk) {
    const I row=pos[0]++,node=m[row*13+1];
    if(node<0||node>=nodes){status[0]=2;break;}
    if(!((__gm__ uint8_t*)connected)[row]||k[node]!=target)continue;
    const I owner=map[node];if(!m[row*13+3]||owner<0||owner>=parameter_count){status[0]=2;break;}
    s[used]=row;p[used]=owner;d[used]=row;++used;
    ((__gm__ uint8_t*)content_connected)[row]=residual;
    ((__gm__ uint8_t*)comparison_connected)[row]=1;
    if(residual||(target-1)%3)((__gm__ uint8_t*)parameter_connected)[node]=1;
    bool found=false;for(I j=0;j<n[0];++j)found|=o[j]==owner;if(!found)o[n[0]++]=owner;
  }
  if(!status[0]&&used){go[0]=1;++((__gm__ I*)chunks)[0];}
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
