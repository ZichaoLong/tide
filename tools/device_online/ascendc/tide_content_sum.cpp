#include "kernel_operator.h"
namespace {
using I=int64_t;
}
// Initial exact-order packed implementation. A single task handles actual
// ragged fibers, never host-per-message placement. Scalar AIV numerical loops
// are a correctness implementation, not a throughput optimization claim.
extern "C" __global__ __aicore__ void tide_content_sum(GM_ADDR coordinates,GM_ADDR values,
    GM_ADDR offsets,GM_ADDR fibers,GM_ADDR lengths,GM_ADDR sources,GM_ADDR scales,
    GM_ADDR content,GM_ADDR weighted,GM_ADDR error,int64_t capacity,int64_t width,int64_t nodes,int64_t inputs,int64_t edges) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)coordinates);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto c=(__gm__ I*)coordinates,o=(__gm__ I*)offsets,f=(__gm__ I*)fibers;
  auto src=(__gm__ I*)sources;auto x=(__gm__ float*)values,w=(__gm__ float*)scales;
  auto h=(__gm__ float*)content,z=(__gm__ float*)weighted;
  I count=((__gm__ I*)lengths)[1];
  if(status[0]==0&&(count<0||count>capacity))status[0]=2;
  for(I i=0;i<count&&status[0]==0;++i) {
    I node=f[i*4+1],first=o[i],end=o[i+1];
    if(node<0||node>=nodes||first<0||end<=first||end>capacity){status[0]=2;break;}
    for(I a=first;a<end&&status[0]==0;++a) {
      I kind=c[a*6+3],id=c[a*6+4];
      if((kind!=0&&kind!=1)||id<0||id>=(kind==0?inputs:edges)){status[0]=2;break;}
      I key=id+(kind==0?0:inputs);
      if(src[key*2]!=node){status[0]=2;break;}
      for(I prev=first;prev<a;++prev) {
        I pkey=c[prev*6+4]+(c[prev*6+3]==0?0:inputs);
        if(src[pkey*2+1]==src[key*2+1]){status[0]=2;break;}
      }
      for(I j=0;j<width;++j)z[a*width+j]=x[a*width+j]*w[key];
    }
    if(status[0])break;
    for(I j=0;j<width;++j) {
      float sum=z[first*width+j];
      for(I a=first+1;a<end;++a)sum=sum+z[a*width+j];
      h[i*width+j]=sum;
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
