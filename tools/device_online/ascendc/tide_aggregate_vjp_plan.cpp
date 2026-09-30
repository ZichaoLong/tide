#include "kernel_operator.h"
extern "C" __global__ __aicore__ void tide_aggregate_vjp_plan(GM_ADDR metadata,GM_ADDR count,GM_ADDR range,
    GM_ADDR connected,GM_ADDR consumer_head,GM_ADDR consumer_next,GM_ADDR links,GM_ADDR sources,
    GM_ADDR kinds,GM_ADDR lengths,GM_ADDR weights,GM_ADDR cursor,GM_ADDR ids,GM_ADDR row_nodes,
    GM_ADDR message_ids,GM_ADDR owners,GM_ADDR owner_count,GM_ADDR logits,GM_ADDR probabilities,
    GM_ADDR parameter_connected,GM_ADDR branch,GM_ADDR chunks,GM_ADDR error,
    int64_t capacity,int64_t fibers,int64_t physical,int64_t nodes,int64_t slots,int64_t chunk,int64_t target) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);using I=int64_t;
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto e=(__gm__ I*)metadata,pos=(__gm__ I*)cursor,ix=(__gm__ I*)ids,rn=(__gm__ I*)row_nodes;
  auto head=(__gm__ I*)consumer_head,next=(__gm__ I*)consumer_next,m=(__gm__ I*)links,s=(__gm__ I*)sources;
  auto k=(__gm__ I*)kinds,length=(__gm__ I*)lengths,mi=(__gm__ I*)message_ids,owner=(__gm__ I*)owners;
  auto w=(__gm__ float*)weights,l=(__gm__ float*)logits,p=(__gm__ float*)probabilities;
  auto flags=(__gm__ uint8_t*)parameter_connected;auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;
  const I rows=((__gm__ I*)count)[0],first=((__gm__ I*)range)[0];
  I used=0,unique=0;go[0]=0;((__gm__ I*)owner_count)[0]=0;
  for(I r=0;r<chunk;++r) {
    ix[r]=-1;rn[r]=-1;owner[r]=-1;
    for(I j=0;j<slots;++j){mi[r*slots+j]=-1;p[r*slots+j]=0.f;
      l[r*slots+j]=j?AscendC::GetScalarBitcodeValue<uint32_t,float>(0xff800000):0.f;}
  }
  if(status[0])return;
  if(rows<0||rows>capacity||first<0||first>capacity-rows||pos[0]<0||pos[0]>rows){status[0]=2;return;}
  while(pos[0]<rows&&used<chunk) {
    const I row=pos[0]++,node=e[(first+row)*13+1];
    if(node<0||node>=nodes){status[0]=2;break;}
    if(!((__gm__ uint8_t*)connected)[row]||k[node]!=target)continue;
    const I n=length[node];if(n<1||n>slots){status[0]=2;break;}
    ix[used]=row;rn[used]=node;
    for(I j=0;j<slots;++j)l[used*slots+j]=AscendC::GetScalarBitcodeValue<uint32_t,float>(0xff800000);
    if(target==4)for(I j=0;j<n;++j){l[used*slots+j]=w[node*slots+j];flags[node*slots+j]=1;}
    I messages=0;
    for(I a=head[first+row];a>=0;a=next[a]) {
      if(a>=fibers||++messages>fibers){status[0]=2;break;}
      const I source=m[a*4+2];if(source<0||source>=physical||s[source*2]!=node){status[0]=2;break;}
      const I slot=s[source*2+1];if(slot<0||slot>=n||mi[used*slots+slot]>=0){status[0]=2;break;}
      mi[used*slots+slot]=a;
      if(target==2||target==3){l[used*slots+slot]=w[node*slots+slot];flags[node*slots+slot]=1;}
    }
    if(status[0]||!messages){status[0]=2;break;}
    if(target==1)for(I j=0;j<n;++j)if(mi[used*slots+j]>=0)p[used*slots+j]=1.f/static_cast<float>(messages);
    bool seen=false;for(I j=0;j<unique;++j)seen|=owner[j]==node;if(!seen)owner[unique++]=node;
    ++used;
  }
  if(!status[0]&&used){go[0]=1;((__gm__ I*)owner_count)[0]=unique;++((__gm__ I*)chunks)[0];}
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
