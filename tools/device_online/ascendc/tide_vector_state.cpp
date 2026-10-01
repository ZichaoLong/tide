#include "kernel_operator.h"
#include "state_clock.h"
#include "state_payload.h"
namespace {
using I=int64_t;
constexpr uint32_t tile=256;
template<class T>
class VectorState {
 public:
  __aicore__ inline void run(GM_ADDR lengths,GM_ADDR content,GM_ADDR config,GM_ADDR coefficients,
      GM_ADDR retention,GM_ADDR policy,GM_ADDR state,GM_ADDR comparisons,GM_ADDR metadata,GM_ADDR diagnostics,
      GM_ADDR proposals,GM_ADDR error,I width,I nodes,bool trace) {
    AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
    AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
    if(((__gm__ int32_t*)error)[0])return;
    auto events=(__gm__ I*)metadata,cfg=(__gm__ I*)config;
    auto rho=(__gm__ T*)retention;
    auto h=(__gm__ T*)content,a=(__gm__ T*)coefficients,s=(__gm__ T*)state;
    auto c=(__gm__ T*)comparisons,ap=(__gm__ T*)proposals;auto v=(__gm__ float*)diagnostics;
    io.init(pipe);
    pipe.InitBuffer(old_buffer,tile*sizeof(float));pipe.InitBuffer(proposal_buffer,tile*sizeof(float));
    pipe.InitBuffer(summary_buffer,tile*sizeof(float));pipe.InitBuffer(coefficient_buffer,tile*sizeof(float));
    auto old=old_buffer.Get<float>(),proposal=proposal_buffer.Get<float>();
    auto summary=summary_buffer.Get<float>(),coefficient=coefficient_buffer.Get<float>();
    const I count=((__gm__ I*)lengths)[1],tiles=(width+tile-1)/tile,stride=5*width+2;
    // Metadata preflight establishes sorted owners and increasing int64 clocks.
    // Each owner/tile has one writer; its actual time sequence stays in order.
    for(I task=AscendC::GetBlockIdx();task<count*tiles;task+=AscendC::GetBlockNum()) {
      const I first=task/tiles,start=(task%tiles)*tile;
      const I b=events[first*13],n=events[first*13+1],key=b*nodes+n,kind=cfg[n*3];
      if(first&&events[(first-1)*13]==b&&events[(first-1)*13+1]==n)continue;
      const uint32_t size=width-start<tile?width-start:tile;
      io.load(old,s,key*width+start,size);
      if(kind==1)io.load(coefficient,a,n*width+start,size);
      for(I i=first;i<count&&events[i*13]==b&&events[i*13+1]==n;++i) {
        const bool active=events[i*13+3],adopt=cfg[n*3+2]||active,clear=cfg[n*3+1]&&active;
        if(kind||trace)io.load(summary,h,i*width+start,size);
        if(trace){io.save(summary,v,i*stride+start,size);io.save(old,v,i*stride+width+start,size);}
        AscendC::Muls(proposal,old,1.0f,size);AscendC::PipeBarrier<PIPE_V>();
        if(kind==1) {
          AscendC::Mul(proposal,old,coefficient,size);AscendC::PipeBarrier<PIPE_V>();io.round<T>(proposal,size);
          AscendC::Add(proposal,proposal,summary,size);AscendC::PipeBarrier<PIPE_V>();
        } else if(kind==2) {
          const float retention_value=rho[n];
          const auto local=tide_device::local_time_unchecked(events[i*13+2],(__gm__ I*)policy,n);
          const auto local_old=tide_device::local_time_unchecked(events[i*13+4],(__gm__ I*)policy,n);
          const uint64_t ticks=(uint64_t(local)+1)-uint64_t(local_old+1);
          for(uint64_t tick=0;tick<ticks;++tick) {
            AscendC::Muls(proposal,proposal,retention_value,size);AscendC::PipeBarrier<PIPE_V>();io.round<T>(proposal,size);
          }
          AscendC::Add(proposal,summary,proposal,size);AscendC::PipeBarrier<PIPE_V>();
        }
        io.round<T>(proposal,size);
        if(kind==3)io.load(proposal,ap,i*width+start,size);
        if(trace)io.save(proposal,v,i*stride+2*width+start,size);
        if(adopt){AscendC::Muls(old,proposal,1.0f,size);AscendC::PipeBarrier<PIPE_V>();}
        io.save(old,c,i*width+start,size);
        if(trace)io.save(old,v,i*stride+3*width+start,size);
        if(clear){AscendC::Muls(old,old,0.0f,size);AscendC::PipeBarrier<PIPE_V>();}
        if(trace)io.save(old,v,i*stride+4*width+start,size);
      }
      io.save(old,s,key*width+start,size);
    }
  }
 private:
  AscendC::TPipe pipe;
  tide_device::StatePayload io;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> old_buffer,proposal_buffer,summary_buffer,coefficient_buffer;
};
}
extern "C" __global__ __aicore__ void tide_vector_state(GM_ADDR lengths,GM_ADDR content,
    GM_ADDR config,GM_ADDR coefficients,GM_ADDR retention,GM_ADDR policy,GM_ADDR state,GM_ADDR comparisons,
    GM_ADDR metadata,GM_ADDR diagnostics,GM_ADDR proposals,GM_ADDR error,int64_t width,int64_t nodes,int64_t trace,int64_t fp16) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(fp16){VectorState<half> op;op.run(lengths,content,config,coefficients,retention,policy,state,comparisons,metadata,diagnostics,proposals,error,width,nodes,trace);}
  else {VectorState<float> op;op.run(lengths,content,config,coefficients,retention,policy,state,comparisons,metadata,diagnostics,proposals,error,width,nodes,trace);}
}
