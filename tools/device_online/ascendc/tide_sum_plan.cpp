#include "kernel_operator.h"
namespace {
using I=int64_t;
// The physical table stays untouched. Only Aggregate's view is projected.
__aicore__ inline bool before(__gm__ I* c,__gm__ I* origins,I a,I b) {
  I ak=c[a*6+3],aid=c[a*6+4],ap=c[a*6+5];
  I bk=c[b*6+3],bid=c[b*6+4],bp=c[b*6+5];
  if(ak==1&&origins[aid*2]>=0){ak=0;ap/=origins[aid*2+1];aid=origins[aid*2];}
  if(bk==1&&origins[bid*2]>=0){bk=0;bp/=origins[bid*2+1];bid=origins[bid*2];}
  return ak!=bk?ak<bk:aid!=bid?aid<bid:ap<bp;
}
}
// Single metadata preflight before parallel numerical tasks. No payload reads
// or persistent commits occur when a group is malformed.
extern "C" __global__ __aicore__ void tide_sum_plan(GM_ADDR coordinates,GM_ADDR offsets,
    GM_ADDR fibers,GM_ADDR lengths,GM_ADDR sources,GM_ADDR origins,GM_ADDR keys,GM_ADDR order,GM_ADDR error,
    int64_t capacity,int64_t nodes,int64_t inputs,int64_t edges,int64_t project) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)coordinates);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto c=(__gm__ I*)coordinates,o=(__gm__ I*)offsets;
  auto f=(__gm__ I*)fibers,src=(__gm__ I*)sources,k=(__gm__ I*)keys,counts=(__gm__ I*)lengths;
  auto view=(__gm__ I*)origins,ordered=(__gm__ I*)order;
  const I atoms=counts[0],count=counts[1];
  if(status[0]==0&&(atoms<0||atoms>capacity||count<0||count>capacity||o[0]!=0))status[0]=2;
  for(I i=0;i<count&&status[0]==0;++i) {
    const I node=f[i*4+1],first=o[i],end=o[i+1];
    if(node<0||node>=nodes||first<0||end<=first||end>atoms){status[0]=2;break;}
    for(I a=first;a<end;++a) {
      const I kind=c[a*6+3],id=c[a*6+4];
      if((kind!=0&&kind!=1)||id<0||id>=(kind==0?inputs:edges)
          ||c[a*6]!=f[i*4]||c[a*6+1]!=node||c[a*6+2]!=f[i*4+2]){status[0]=2;break;}
      const I key=id+(kind==0?0:inputs);
      if(src[key*2]!=node){status[0]=2;break;}
      if(project&&kind==1) {
        const I port=view[id*2],stride=view[id*2+1];
        if(port< -1||stride<1){status[0]=2;break;}
        if(port>=0&&c[a*6+5]%stride){status[0]=10;break;}
      }
      for(I prev=first;prev<a;++prev)
        if(src[k[prev]*2+1]==src[key*2+1]){status[0]=2;break;}
      if(status[0])break;
      k[a]=key;ordered[a]=a;
    }
    // Stable insertion sorts actual fiber metadata, never payload vectors.
    // No numerical host prepass or changes to physical message identity.
    if(project&&status[0]==0)for(I a=first+1;a<end;++a) {
      I row=ordered[a],pos=a;
      while(pos>first&&before(c,view,row,ordered[pos-1])){ordered[pos]=ordered[pos-1];--pos;}
      ordered[pos]=row;
    }
  }
  if(status[0]==0&&o[count]!=atoms)status[0]=2;
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
