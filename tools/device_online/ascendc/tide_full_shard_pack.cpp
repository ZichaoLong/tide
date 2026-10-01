#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_full_shard_pack(GM_ADDR coordinates,GM_ADDR valid,GM_ADDR mapping,
    GM_ADDR local_coordinates,GM_ADDR local_valid,GM_ADDR sources,GM_ADDR destinations,GM_ADDR branch,
    GM_ADDR work,GM_ADDR error,int64_t rows,int64_t nodes,int64_t local_nodes) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)coordinates);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto c=(__gm__ I*)coordinates,map=(__gm__ I*)mapping,out=(__gm__ I*)local_coordinates;
  auto source=(__gm__ I*)sources,dest=(__gm__ I*)destinations,stats=(__gm__ I*)work;
  auto live=(__gm__ uint8_t*)valid,selected=(__gm__ uint8_t*)local_valid;
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;go[0]=0;
  for(I i=0;i<rows;++i){source[i]=rows;dest[i]=rows+i;selected[i]=0;for(I j=0;j<4;++j)out[i*4+j]=0;}
  I used=0;
  for(I row=0;status[0]==0&&row<rows;++row)if(live[row]) {
    const I n=c[row*4+1];if(n<0||n>=nodes){status[0]=2;break;}
    const I local=map[n];if(local==-1)continue;
    if(local<0||local>=local_nodes){status[0]=2;break;}
    source[used]=dest[used]=row;selected[used]=1;
    for(I j=0;j<4;++j)out[used*4+j]=c[row*4+j];out[used*4+1]=local;++used;
  }
  if(status[0]==0&&used) {
    if(stats[0]>I(0x7fffffffffffffff)-used||stats[1]>I(0x7fffffffffffffff)-rows)status[0]=5;
    else{stats[0]+=used;stats[1]+=rows;go[0]=1;}
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
