#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_projection_shard_plan(GM_ADDR global_rows,
    GM_ADDR mapping,GM_ADDR sources,GM_ADDR parameters,GM_ADDR destinations,GM_ADDR owners,
    GM_ADDR owner_count,GM_ADDR connected,GM_ADDR branch,GM_ADDR error,
    int64_t global_count,int64_t local_count,int64_t chunk,int64_t reverse) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)global_rows);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto input=(__gm__ I*)global_rows,map=(__gm__ I*)mapping,src=(__gm__ I*)sources;
  auto param=(__gm__ I*)parameters,dst=(__gm__ I*)destinations,list=(__gm__ I*)owners;
  auto n=(__gm__ I*)owner_count;auto go=(__gm__ int32_t*)branch,status=(__gm__ int32_t*)error;
  go[0]=0;n[0]=0;
  for(I j=0;j<chunk;++j){src[j]=chunk;param[j]=local_count;dst[j]=chunk+j;list[j]=-1;}
  I used=0;
  for(I j=0;status[0]==0&&j<chunk;++j) {
    const I global=input[j];if(global==global_count)continue;
    if(global<0||global>=global_count){status[0]=2;break;}
    const I local=map[global];if(local==-1)continue;
    if(local<0||local>=local_count){status[0]=2;break;}
    src[used]=j;param[used]=local;dst[used]=j;++used;
    if(reverse)((__gm__ uint8_t*)connected)[local]=1;
    bool seen=false;for(I k=0;k<n[0];++k)seen|=list[k]==local;
    if(!seen)list[n[0]++]=local;
  }
  if(status[0]==0&&used)go[0]=1;
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
