#include "kernel_operator.h"
#include "state_clock.h"
namespace {
using I=int64_t;
template<class T>
__aicore__ inline void content_state(GM_ADDR fibers,GM_ADDR lengths,
    GM_ADDR content,GM_ADDR scores,GM_ADDR controls,GM_ADDR active,GM_ADDR config,GM_ADDR coefficients,GM_ADDR retention,GM_ADDR policy,
    GM_ADDR state,GM_ADDR clocks,GM_ADDR present,GM_ADDR action_coordinates,GM_ADDR comparisons,
    GM_ADDR event_meta,GM_ADDR event_values,GM_ADDR stage,GM_ADDR event_count,GM_ADDR proposals,GM_ADDR error,
    int64_t capacity,int64_t width,int64_t nodes,int64_t samples,int64_t diagnostics,int64_t metadata_only,int64_t max_ticks) {
  if(AscendC::GetBlockIdx()!=0)return;
  AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)clocks);
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  auto status=(__gm__ int32_t*)error;auto fs=(__gm__ I*)fibers,cfg=(__gm__ I*)config,t=(__gm__ I*)clocks;
  auto s=(__gm__ T*)state,h=(__gm__ T*)content,a=(__gm__ T*)coefficients;
  auto valid=(__gm__ uint8_t*)present,on=(__gm__ uint8_t*)active;
  auto events=(__gm__ I*)event_meta,actions=(__gm__ I*)action_coordinates;auto v=(__gm__ float*)event_values;
  I count=((__gm__ I*)lengths)[1],stride=5*width+2;
  auto total=(__gm__ I*)event_count;
  if(status[0]==0&&(count<0||count>capacity))status[0]=2;
  if(status[0]==0&&(total[0]<0||count>I(0x7fffffffffffffff)-total[0]))status[0]=5;
  for(I i=0;i<count&&status[0]==0;++i) {
    I b=fs[i*4],n=fs[i*4+1],time=fs[i*4+2];
    if(b<0||b>=samples||n<0||n>=nodes||time<0){status[0]=2;break;}
    I key=b*nodes+n,old_time=t[key*2],old_count=t[key*2+1];
    const I kind=cfg[n*3];bool clear=cfg[n*3+1]&&on[i],adopt=cfg[n*3+2]||on[i];
    if(old_count<0||old_time< -1||old_time>=time||kind<0||kind>3){status[0]=2;break;}
    if(i&&(b<fs[(i-1)*4]||(b==fs[(i-1)*4]&&(n<fs[(i-1)*4+1]
        ||(n==fs[(i-1)*4+1]&&time<=fs[(i-1)*4+2]))))){status[0]=2;break;}
    if(kind&&old_count==I(0x7fffffffffffffff)){status[0]=5;break;}
    I local,local_old;
    if(!tide_device::local_time(time,(__gm__ I*)policy,n,local)
        ||!tide_device::local_time(old_time,(__gm__ I*)policy,n,local_old)){status[0]=9;break;}
    const uint64_t ticks=(uint64_t(local)+1)-uint64_t(local_old+1);
    if(kind==2&&ticks>uint64_t(max_ticks)){status[0]=8;break;}
    I proposed_time=kind?time:old_time,proposed_count=old_count+(kind?1:0);
    I next_time=adopt?proposed_time:old_time,next_count=adopt?proposed_count:old_count;
    if(diagnostics||metadata_only) {
      auto row=events+i*13;row[12]=((__gm__ I*)stage)[0];
      row[0]=b;row[1]=n;row[2]=time;row[3]=on[i];row[4]=old_time;row[5]=old_count;
      row[6]=proposed_time;row[7]=proposed_count;row[8]=next_time;row[9]=next_count;row[10]=next_time;row[11]=next_count;
    }
    for(I j=0;j<width&&!metadata_only;++j) {
      float old=s[key*width+j],proposal=old;
      if(kind==1){float decayed=float(T(float(a[n*width+j])*old));proposal=float(T(decayed+float(h[i*width+j])));}
      if(kind==2){float rho=float(((__gm__ T*)retention)[n]);
        for(uint64_t tick=0;tick<ticks;++tick)proposal=float(T(proposal*rho));
        proposal=float(T(float(h[i*width+j])+proposal));}
      if(kind==3)proposal=float(((__gm__ T*)proposals)[i*width+j]);
      float comparison=adopt?proposal:old,next=clear?comparison*0.0f:comparison;
      if(diagnostics) {
        v[i*stride+j]=h[i*width+j];v[i*stride+width+j]=old;v[i*stride+2*width+j]=proposal;
        v[i*stride+3*width+j]=comparison;v[i*stride+4*width+j]=next;
      }
      s[key*width+j]=T(next);
      ((__gm__ T*)comparisons)[i*width+j]=T(comparison);
    }
    if(diagnostics){v[i*stride+5*width]=((__gm__ float*)scores)[i];v[i*stride+5*width+1]=((__gm__ float*)controls)[i];}
    t[key*2]=next_time;t[key*2+1]=next_count;valid[key]=1;
    actions[i*4]=b;actions[i*4+1]=n;actions[i*4+2]=time;actions[i*4+3]=time;
  }
  if(status[0]==0)total[0]+=count;
  AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
}
} // namespace
extern "C" __global__ __aicore__ void tide_content_state(GM_ADDR fibers,GM_ADDR lengths,
    GM_ADDR content,GM_ADDR scores,GM_ADDR controls,GM_ADDR active,GM_ADDR config,GM_ADDR coefficients,GM_ADDR retention,GM_ADDR policy,
    GM_ADDR state,GM_ADDR clocks,GM_ADDR present,GM_ADDR action_coordinates,GM_ADDR comparisons,
    GM_ADDR event_meta,GM_ADDR event_values,GM_ADDR stage,GM_ADDR event_count,GM_ADDR proposals,GM_ADDR error,
    int64_t capacity,int64_t width,int64_t nodes,int64_t samples,int64_t diagnostics,int64_t metadata_only,int64_t max_ticks,int64_t fp16) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(fp16)content_state<half>(fibers,lengths,content,scores,controls,active,config,coefficients,retention,policy,
    state,clocks,present,action_coordinates,comparisons,event_meta,event_values,stage,event_count,proposals,error,
    capacity,width,nodes,samples,diagnostics,metadata_only,max_ticks);
  else content_state<float>(fibers,lengths,content,scores,controls,active,config,coefficients,retention,policy,
    state,clocks,present,action_coordinates,comparisons,event_meta,event_values,stage,event_count,proposals,error,
    capacity,width,nodes,samples,diagnostics,metadata_only,max_ticks);
}
