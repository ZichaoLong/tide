#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_emission_vjp_plan(GM_ADDR messages,GM_ADDR heads,GM_ADDR next,
    GM_ADDR mapping,GM_ADDR connected,GM_ADDR range,GM_ADDR cursor,GM_ADDR sources,GM_ADDR events,
    GM_ADDR parameters,GM_ADDR destinations,GM_ADDR owners,GM_ADDR owner_count,GM_ADDR parameter_connected,
    GM_ADDR branch,GM_ADDR chunks,GM_ADDR error,int64_t capacity,int64_t total,int64_t projections,int64_t chunk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)messages);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto h=(__gm__ I*)heads,n=(__gm__ I*)next,map=(__gm__ I*)mapping,position=(__gm__ I*)cursor;
  auto source=(__gm__ I*)sources,event=(__gm__ I*)events,param=(__gm__ I*)parameters,dest=(__gm__ I*)destinations;
  auto list=(__gm__ I*)owners,nowners=(__gm__ I*)owner_count,on=(__gm__ I*)range;
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;go[0]=0;nowners[0]=0;
  const I first=on[0],end=on[1];
  for(I i=0;i<chunk;++i){source[i]=total;event[i]=capacity;param[i]=projections;dest[i]=total+i;list[i]=-1;}
  if(status[0])return;
  if(first<0||end<first||end>capacity){status[0]=2;return;}
  if(position[0]<0){position[0]=first;position[1]=first<end?h[first]:-1;}
  I used=0;
  while(position[0]<end&&used<chunk) {
    const I m=position[1];
    if(m<0){++position[0];if(position[0]<end)position[1]=h[position[0]];continue;}
    if(m>=total||map[m]>=projections){status[0]=2;break;}
    position[1]=n[m];if(!((__gm__ uint8_t*)connected)[m]||map[m]<0)continue;
    const I owner=map[m];source[used]=m;event[used]=position[0];param[used]=owner;dest[used]=m;++used;
    ((__gm__ uint8_t*)parameter_connected)[owner]=1;
    bool found=false;for(I j=0;j<nowners[0];++j)found|=list[j]==owner;
    if(!found)list[nowners[0]++]=owner;
  }
  if(status[0]==0&&used){go[0]=1;++((__gm__ I*)chunks)[0];}
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
