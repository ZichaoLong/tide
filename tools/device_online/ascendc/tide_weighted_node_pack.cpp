#include "kernel_operator.h"
// Actual complete fibers -> private sample/node rows, with one physical atom
// per slot. SrcClass chooses parameters; repeated classes never coalesce.
extern "C" __global__ __aicore__ void tide_weighted_node_pack(GM_ADDR fibers,GM_ADDR offsets,GM_ADDR coordinates,
    GM_ADDR counts,GM_ADDR classes,GM_ADDR indices,GM_ADDR owners,GM_ADDR lengths,GM_ADDR times,GM_ADDR present,
    GM_ADDR bank,GM_ADDR error,int64_t node,int64_t samples,int64_t capacity,int64_t inputs,int64_t sources,int64_t slots){
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)indices);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto ix=(__gm__ I*)indices,owner=(__gm__ I*)owners,n=(__gm__ I*)lengths,t=(__gm__ I*)times,banks=(__gm__ I*)bank;
  auto live=(__gm__ uint8_t*)present;auto status=(__gm__ int32_t*)error;
  for(I b=0;b<samples;++b){n[b]=t[b]=0;live[b]=0;banks[b]=-1;}
  for(I i=0;i<samples*slots;++i){ix[i]=capacity;owner[i]=-1;}
  auto f=(__gm__ I*)fibers,off=(__gm__ I*)offsets,c=(__gm__ I*)coordinates,labels=(__gm__ I*)classes;
  const I nf=((__gm__ I*)counts)[1];
  if(!status[0]&&(nf<0||nf>capacity)){status[0]=2;}
  if(!status[0])for(I k=0;k<nf;++k)if(f[k*4+1]==node){
    I b=f[k*4],a=off[k],end=off[k+1];
    if(b<0||b>=samples||live[b]||a<0||end<=a||end>capacity||end-a>slots){status[0]=2;break;}
    live[b]=1;banks[b]=0;t[b]=f[k*4+2];n[b]=end-a;
    for(I j=a;j<end;++j){I kind=c[j*6+3],physical=c[j*6+4],source=physical+(kind==1?inputs:0);
      if((kind!=0&&kind!=1)||physical<0||source<0||source>=sources||(kind==0&&physical>=inputs)||labels[source*5]<0){status[0]=2;break;}
      auto meta=labels+source*5;
      if(meta[4]<1||c[j*6+5]%meta[4]){status[0]=2;break;}
      I pos=j-a;
      // SourceDomain exclusivity and SrcClass multiplicity remain distinct.
      for(I k=0;k<j-a;++k){I prev=ix[b*slots+k],src=c[prev*6+4]+(c[prev*6+3]==1?inputs:0);
        if(labels[src*5+1]==meta[1]){status[0]=2;break;}}
      if(status[0])break;
      while(pos>0){I prev=ix[b*slots+pos-1],src=c[prev*6+4]+(c[prev*6+3]==1?inputs:0);auto pm=labels+src*5;
        bool before=meta[2]<pm[2]||(meta[2]==pm[2]&&(meta[3]<pm[3]||(meta[3]==pm[3]&&c[j*6+5]/meta[4]<c[prev*6+5]/pm[4])));
        if(!before)break;ix[b*slots+pos]=prev;owner[b*slots+pos]=owner[b*slots+pos-1];--pos;}
      ix[b*slots+pos]=j;owner[b*slots+pos]=meta[0];
    }
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
