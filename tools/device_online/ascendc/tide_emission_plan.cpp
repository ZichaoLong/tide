#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_emission_plan(GM_ADDR actions,GM_ADDR active,
    GM_ADDR offsets,GM_ADDR periods,GM_ADDR slots,GM_ADDR meta,GM_ADDR count,GM_ADDR sources,
    GM_ADDR edge_coords,GM_ADDR edge_valid,GM_ADDR edge_sources,GM_ADDR edge_scales,
    GM_ADDR output_coords,GM_ADDR output_valid,GM_ADDR output_sources,GM_ADDR output_scales,
    GM_ADDR branch,GM_ADDR error,int64_t rows,int64_t nodes,int64_t samples,int64_t slot_count,
    int64_t capacity,int64_t edge_capacity,int64_t output_capacity) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)meta);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto ac=(__gm__ I*)actions,off=(__gm__ I*)offsets,period=(__gm__ I*)periods,table=(__gm__ I*)slots;
  auto em=(__gm__ I*)meta,total=(__gm__ I*)count,source=(__gm__ I*)sources;
  auto ec=(__gm__ I*)edge_coords,es=(__gm__ I*)edge_sources,ew=(__gm__ I*)edge_scales;
  auto oc=(__gm__ I*)output_coords,os=(__gm__ I*)output_sources,ow=(__gm__ I*)output_scales;
  auto on=(__gm__ uint8_t*)active,ev=(__gm__ uint8_t*)edge_valid,ov=(__gm__ uint8_t*)output_valid;
  auto go=(__gm__ int32_t*)branch,status=(__gm__ int32_t*)error;go[0]=0;total[0]=0;
  // Initialize only metadata here. No tensor arithmetic occurs before complete
  // capacity/time preflight, and poisoned absent values never enter a gather.
  for(I i=0;i<capacity;++i){source[i]=rows;for(I j=0;j<6;++j)em[i*6+j]=0;}
  for(I i=0;i<edge_capacity;++i){ev[i]=0;es[i]=capacity;ew[i]=slot_count;}
  for(I i=0;i<output_capacity;++i){ov[i]=0;os[i]=capacity;ow[i]=slot_count;}
  I edges=0,outputs=0,emissions=0;
  for(I row=0;row<rows&&status[0]==0;++row)if(on[row]) {
    auto a=ac+row*4;const I node=a[1];
    if(a[0]<0||a[0]>=samples||node<0||node>=nodes||a[2]<0||a[3]<0){status[0]=2;break;}
    for(I j=off[node];j<off[node+1];++j) {
      auto s=table+j*6;
      if(s[0]==-2||(s[0]>=0&&a[2]%period[node]!=s[0]))continue;
      if(emissions==capacity||(s[1]==1?edges==edge_capacity:outputs==output_capacity)){status[0]=1;break;}
      if(s[1]==1&&s[4]>I(0x7fffffffffffffff)-a[2]){status[0]=3;break;}
      ++emissions;if(s[1]==1)++edges;else ++outputs;
    }
  }
  if(status[0]==0) {
    I cursor=0,edge=0,output=0;
    for(I row=0;row<rows;++row)if(on[row]) {
      auto a=ac+row*4;const I node=a[1];
      for(I j=off[node];j<off[node+1];++j) {
        auto s=table+j*6;
        if(s[0]==-2||(s[0]>=0&&a[2]%period[node]!=s[0]))continue;
        auto m=em+cursor*6;m[0]=a[0];m[1]=node;m[2]=a[2];m[3]=a[3];m[4]=j-off[node];m[5]=s[5];
        source[cursor]=row;
        auto dest=s[1]==1?ec+edge*6:oc+output*6;
        dest[0]=a[0];dest[1]=s[3];dest[2]=a[2]+s[4];dest[3]=s[1];dest[4]=s[2];dest[5]=a[3];
        if(s[1]==1){ev[edge]=1;es[edge]=cursor;ew[edge]=j;++edge;}
        else {ov[output]=1;os[output]=cursor;ow[output]=j;++output;}
        ++cursor;
      }
    }
    total[0]=cursor;go[0]=cursor>0;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
