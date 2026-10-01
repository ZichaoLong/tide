#include "kernel_operator.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_event_reverse_plan(GM_ADDR events,GM_ADDR config,
    GM_ADDR ranges,GM_ADDR previous,GM_ADDR tails,GM_ADDR stage,GM_ADDR proposals,
    GM_ADDR key_connected,GM_ADDR value_connected,GM_ADDR plan,GM_ADDR flags,GM_ADDR lengths,GM_ADDR branch,GM_ADDR error,
    int64_t owners,int64_t parameters,int64_t chunk) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)events);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error,go=(__gm__ int32_t*)branch;go[0]=0;
  auto e=(__gm__ I*)events,cfg=(__gm__ I*)config,r=(__gm__ I*)ranges,prev=(__gm__ I*)previous,tail=(__gm__ I*)tails;
  auto interval=(__gm__ I*)stage,rows=(__gm__ I*)plan,lens=(__gm__ I*)lengths;
  auto flag=(__gm__ uint8_t*)flags,kc=(__gm__ uint8_t*)key_connected,vc=(__gm__ uint8_t*)value_connected;
  for(I i=0;i<chunk;++i){for(I j=0;j<8;++j)rows[i*8+j]=-1;for(I j=0;j<6;++j)flag[i*6+j]=0;lens[i]=0;}
  if(status[0])return;I count=0;
  for(I owner=0;owner<owners&&count<chunk;++owner) {
    const I event=tail[owner];if(event<interval[0])continue;
    if(event>=interval[1]){status[0]=2;break;}
    const I param=owner%parameters,local=event-interval[0];auto row=rows+count*8;
    row[0]=event;row[1]=owner;row[2]=param;row[3]=local;for(I j=0;j<4;++j)row[4+j]=r[event*4+j];
    const bool active=e[event*13+3],adopt=cfg[param*2]||active,prop=((__gm__ uint8_t*)proposals)[local];
    auto f=flag+count*6;f[0]=prop;f[1]=prop||(adopt&&kc[owner]);f[2]=prop||(adopt&&vc[owner]);f[3]=prop;
    f[4]=kc[owner];f[5]=vc[owner];
    kc[owner]=(!adopt&&kc[owner])||f[1];vc[owner]=(!adopt&&vc[owner])||f[2];
    lens[count]=prop?row[7]:0;tail[owner]=prev[event];++count;
  }
  if(!status[0]&&count)go[0]=1;
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
