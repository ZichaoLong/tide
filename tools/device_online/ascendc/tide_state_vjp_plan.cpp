#include "kernel_operator.h"
#include "state_clock.h"
namespace {using I=int64_t;}
extern "C" __global__ __aicore__ void tide_state_vjp_plan(GM_ADDR metadata,GM_ADDR count,GM_ADDR config,
    GM_ADDR previous,GM_ADDR tails,GM_ADDR connections,GM_ADDR final_connections,GM_ADDR content_connections,
    GM_ADDR initial_connections,GM_ADDR decay_connections,GM_ADDR retention_connections,GM_ADDR clock_policy,GM_ADDR ticks,
    GM_ADDR proposal_connections,GM_ADDR error,int64_t capacity,int64_t nodes,int64_t samples,int64_t allow_repeat,int64_t max_ticks,int64_t allow_attention) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto m=(__gm__ I*)metadata,cfg=(__gm__ I*)config;
  auto prev=(__gm__ I*)previous,tail=(__gm__ I*)tails;
  const I size=((__gm__ I*)count)[0];
  if(status[0]==0&&(size<0||size>capacity))status[0]=2;
  for(I n=0;n<nodes&&status[0]==0;++n)
    if(cfg[n*3]<0||(cfg[n*3]>3||(cfg[n*3]==2&&!allow_repeat)||(cfg[n*3]==3&&!allow_attention))||(cfg[n*3+1]!=0&&cfg[n*3+1]!=1)||(cfg[n*3+2]!=0&&cfg[n*3+2]!=1))status[0]=12;
  for(I key=0;key<samples*nodes;++key)tail[key]=-1;
  for(I i=0;i<size&&status[0]==0;++i) {
    auto row=m+i*13;const I b=row[0],n=row[1],time=row[2];
    if(b<0||b>=samples||n<0||n>=nodes||time<0||(row[3]!=0&&row[3]!=1)||row[4]<-1||row[4]>=time){status[0]=2;break;}
    const I key=b*nodes+n,last=tail[key];
    if(last>=0&&(m[last*13+2]>=time||row[4]!=m[last*13+10]||row[5]!=m[last*13+11])){status[0]=2;break;}
    const bool adopt=cfg[n*3+2]||row[3];
    const I proposed_time=cfg[n*3]?time:row[4];
    if(row[6]!=proposed_time||row[8]!=(adopt?proposed_time:row[4])||row[10]!=row[8]){status[0]=2;break;}
    ((__gm__ I*)ticks)[i]=0;
    if(cfg[n*3]==2) {
      I local,old;
      if(!tide_device::local_time(time,(__gm__ I*)clock_policy,n,local)
          ||!tide_device::local_time(row[4],(__gm__ I*)clock_policy,n,old)){status[0]=9;break;}
      const uint64_t gap=(uint64_t(local)+1)-uint64_t(old+1);
      if(gap>uint64_t(max_ticks)){status[0]=8;break;}
      ((__gm__ I*)ticks)[i]=I(gap);
    }
    prev[i]=last;tail[key]=i;
  }
  // One metadata writer also owns all byte-sized connection flags. Numerical
  // feature tiles must not race through different bytes of a GM cache line.
  auto flags=(__gm__ uint8_t*)connections,fc=(__gm__ uint8_t*)final_connections;
  auto hc=(__gm__ uint8_t*)content_connections,ic=(__gm__ uint8_t*)initial_connections,dc=(__gm__ uint8_t*)decay_connections;
  auto rc=(__gm__ uint8_t*)retention_connections;
  for(I key=0;key<samples*nodes&&status[0]==0;++key) {
    const I node=key%nodes;bool carry=fc[key],decay=false,retention=false;
    for(I i=tail[key];i>=0;i=prev[i]) {
      const bool adopt=cfg[node*3+2]||m[i*13+3];
      const bool comp=flags[i*5+3]||flags[i*5+4]||carry;
      const bool proposal=flags[i*5+2]||(adopt&&comp);
      hc[i]=flags[i*5]||((cfg[node*3]==1||cfg[node*3]==2)&&proposal);
      if(allow_attention)((__gm__ uint8_t*)proposal_connections)[i]=cfg[node*3]==3&&proposal;
      carry=flags[i*5+1]||(!adopt&&comp)||(cfg[node*3]!=3&&proposal);
      decay|=cfg[node*3]==1&&proposal;
      retention|=cfg[node*3]==2&&proposal&&((__gm__ I*)ticks)[i]>0;
    }
    ic[key]=carry;dc[key]=decay;rc[key]=retention;
  }
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
