#include "kernel_operator.h"
#include "state_payload.h"
namespace {
using I=int64_t;
constexpr uint32_t tile=256;
template<class T> class StateVjp {
 public:
  __aicore__ inline void run(GM_ADDR metadata,GM_ADDR values,GM_ADDR config,GM_ADDR coefficients,
      GM_ADDR previous,GM_ADDR tails,GM_ADDR cotangents,GM_ADDR connections,GM_ADDR final,GM_ADDR final_connections,
      GM_ADDR content,GM_ADDR content_connections,GM_ADDR initial,GM_ADDR initial_connections,
      GM_ADDR decay,GM_ADDR decay_connections,GM_ADDR retention,GM_ADDR ticks,GM_ADDR replay,
      GM_ADDR retention_components,GM_ADDR proposal_gradient,GM_ADDR error,I nodes,I samples,I width,I replay_rows,I scratch_width) {
    AscendC::GlobalTensor<I> cache;cache.SetGlobalBuffer((__gm__ I*)metadata);
    AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
    if(((__gm__ int32_t*)error)[0])return;
    auto m=(__gm__ I*)metadata,cfg=(__gm__ I*)config,prev=(__gm__ I*)previous,tail=(__gm__ I*)tails;
    auto flags=(__gm__ uint8_t*)connections,fc=(__gm__ uint8_t*)final_connections;
    auto v=(__gm__ float*)values,cot=(__gm__ float*)cotangents;
    io.init(pipe);pipe.InitBuffer(input,1,tile*4);pipe.InitBuffer(output,1,tile*4);pipe.InitBuffer(storage,10*tile*4);
    auto space=storage.Get<float>();
    auto carry=space[0],h=space[tile],old=space[2*tile],proposal=space[3*tile],comparison=space[4*tile];
    auto scratch=space[5*tile],a=space[6*tile],dg=space[7*tile],tmp=space[8*tile];
    auto rg=space[9*tile];
    const I tiles=(width+tile-1)/tile,stride=5*width+2;
    // Each owner-feature tile is an independent reverse chain with one writer.
    // Event count and predecessor links were produced on device. Zero-gradient
    // connectivity follows the same branches as values; inactive cotangents
    // and unselected proposals are not evaluated and multiplied by a mask.
    for(I task=AscendC::GetBlockIdx();task<samples*nodes*tiles;task+=AscendC::GetBlockNum()) {
      const I key=task/tiles,n=key%nodes,start=(task%tiles)*tile,kind=cfg[n*3];
      const uint32_t size=width-start<tile?width-start:tile;
      bool carry_on=fc[key];
      read(carry,(__gm__ float*)final,key*width+start,size,carry_on);zero(dg,size);zero(rg,size);
      if(kind==1)io.load(a,(__gm__ T*)coefficients,n*width+start,size);
      for(I i=tail[key];i>=0;i=prev[i]) {
        const bool active=m[i*13+3],adopt=cfg[n*3+2]||active,clear=cfg[n*3+1]&&active;
        const I at=i*5*width+start;
        bool h_on=flags[i*5],old_on=flags[i*5+1],prop_on=flags[i*5+2],comp_on=flags[i*5+3];
        const bool next_on=flags[i*5+4];
        read(h,cot,at,size,h_on);read(old,cot,at+width,size,old_on);
        read(proposal,cot,at+2*width,size,prop_on);read(comparison,cot,at+3*width,size,comp_on);
        if(next_on){load(scratch,cot,at+4*width,size);add(carry,scratch,size);}
        if(carry_on||next_on) {
          if(clear){AscendC::Muls(carry,carry,0.f,size);barrier();}
          add(comparison,carry,size);comp_on=true;
        }
        if(comp_on){if(adopt){add(proposal,comparison,size);prop_on=true;}
          else {add(old,comparison,size);old_on=true;}}
        if(prop_on) {
          if(kind==3)save(proposal,(__gm__ float*)proposal_gradient,i*width+start,size);
          else if(kind==0)add(old,proposal,size);
          else {
            add(h,proposal,size);h_on=true;
            if(kind==1) {
              load(scratch,v,i*stride+width+start,size);
              AscendC::Mul(scratch,scratch,proposal,size);barrier();
              // Match SigmoidBackward's factor before upstream multiplication.
              AscendC::Muls(tmp,a,-1.f,size);barrier();AscendC::Adds(tmp,tmp,1.f,size);barrier();
              AscendC::Mul(tmp,tmp,a,size);barrier();AscendC::Mul(scratch,scratch,tmp,size);barrier();add(dg,scratch,size);
              AscendC::Mul(scratch,proposal,a,size);barrier();add(old,scratch,size);
            } else {
              const float rho=((__gm__ T*)retention)[n];
              AscendC::Muls(tmp,proposal,1.f,size);barrier();
              // Reconstruct literal multiplication inputs in bounded chunks.
              // No division by rho or pow shortcut, including rho=0/negative.
              // Larger chunks reduce prefix recomputation without changing the
              // logical reverse order or truncating a state-clock interval.
              for(I end=((__gm__ I*)ticks)[i];end>0;) {
                const I first=end>replay_rows?end-replay_rows:0;
                load(scratch,v,i*stride+width+start,size);
                for(I tick=0;tick<first;++tick){AscendC::Muls(scratch,scratch,rho,size);barrier();io.round<T>(scratch,size);}
                for(I tick=first;tick<end;++tick) {
                  save(scratch,(__gm__ float*)replay,(I(AscendC::GetBlockIdx())*replay_rows+tick-first)*scratch_width,size);
                  AscendC::Muls(scratch,scratch,rho,size);barrier();io.round<T>(scratch,size);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                for(I tick=end;tick>first;--tick) {
                  load(scratch,(__gm__ float*)replay,(I(AscendC::GetBlockIdx())*replay_rows+tick-first-1)*scratch_width,size);
                  AscendC::Mul(scratch,scratch,tmp,size);barrier();add(rg,scratch,size);
                  AscendC::Muls(tmp,tmp,rho,size);barrier();
                }
                end=first;
              }
              add(old,tmp,size);
            }
          }
          if(kind!=3)old_on=true;
        }
        save(h,(__gm__ float*)content,i*width+start,size);
        AscendC::Muls(carry,old,1.f,size);barrier();carry_on=old_on;
      }
      save(carry,(__gm__ float*)initial,key*width+start,size);save(dg,(__gm__ float*)decay,key*width+start,size);
      save(rg,(__gm__ float*)retention_components,key*width+start,size);
    }
    AscendC::DataCacheCleanAndInvalid<I,AscendC::CacheLine::ENTIRE_DATA_CACHE>(cache);
  }
 private:
  __aicore__ inline void barrier(){AscendC::PipeBarrier<PIPE_V>();}
  __aicore__ inline void zero(AscendC::LocalTensor<float> x,uint32_t size){AscendC::Duplicate(x,0.f,size);barrier();}
  __aicore__ inline void add(AscendC::LocalTensor<float> x,AscendC::LocalTensor<float> y,uint32_t size){AscendC::Add(x,x,y,size);barrier();}
  __aicore__ inline void read(AscendC::LocalTensor<float> x,__gm__ float* from,I offset,uint32_t size,bool on){if(on)load(x,from,offset,size);else zero(x,size);}
  __aicore__ inline void load(AscendC::LocalTensor<float> to,__gm__ float* from,I offset,uint32_t size) {
    AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(from);auto x=input.AllocTensor<float>();
    AscendC::DataCopyPad(x,gm[offset],AscendC::DataCopyExtParams{1,size*4,0,0,0},
      AscendC::DataCopyPadExtParams<float>{true,0,uint8_t((8-size%8)%8),0});
    input.EnQue(x);x=input.DeQue<float>();AscendC::Muls(to,x,1.f,size);barrier();input.FreeTensor(x);
  }
  __aicore__ inline void save(AscendC::LocalTensor<float> from,__gm__ float* to,I offset,uint32_t size) {
    AscendC::GlobalTensor<float> gm;gm.SetGlobalBuffer(to);auto x=output.AllocTensor<float>();AscendC::Muls(x,from,1.f,size);
    output.EnQue(x);x=output.DeQue<float>();AscendC::DataCopyPad(gm[offset],x,AscendC::DataCopyExtParams{1,size*4,0,0,0});output.FreeTensor(x);
  }
  AscendC::TPipe pipe;
  tide_device::StatePayload io;
  AscendC::TQue<AscendC::QuePosition::VECIN,1> input;
  AscendC::TQue<AscendC::QuePosition::VECOUT,1> output;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> storage;
};
}
extern "C" __global__ __aicore__ void tide_state_vjp(GM_ADDR metadata,GM_ADDR values,GM_ADDR config,GM_ADDR coefficients,
    GM_ADDR previous,GM_ADDR tails,GM_ADDR cotangents,GM_ADDR connections,GM_ADDR final,GM_ADDR final_connections,
    GM_ADDR content,GM_ADDR content_connections,GM_ADDR initial,GM_ADDR initial_connections,
    GM_ADDR decay,GM_ADDR decay_connections,GM_ADDR retention,GM_ADDR ticks,GM_ADDR replay,
    GM_ADDR retention_components,GM_ADDR proposal_gradient,GM_ADDR error,int64_t nodes,int64_t samples,int64_t width,int64_t replay_rows,int64_t scratch_width,int64_t fp16) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if(fp16){StateVjp<half> op;op.run(metadata,values,config,coefficients,previous,tails,cotangents,connections,final,final_connections,
    content,content_connections,initial,initial_connections,decay,decay_connections,retention,ticks,replay,
    retention_components,proposal_gradient,error,nodes,samples,width,replay_rows,scratch_width);}
  else {StateVjp<float> op;op.run(metadata,values,config,coefficients,previous,tails,cotangents,connections,final,final_connections,
    content,content_connections,initial,initial_connections,decay,decay_connections,retention,ticks,replay,
    retention_components,proposal_gradient,error,nodes,samples,width,replay_rows,scratch_width);}
}
