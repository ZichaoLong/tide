// Independent CPU consumer of pinned llama.cpp. No Python/LibTorch runtime.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
template<class T> static void save(const std::string & path, const std::vector<T> & values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(T));
    out.close(); require(bool(out), "binary output write failed");
}
template<class T> static std::vector<T> load(const std::string & path) {
    auto size = std::filesystem::file_size(path); require(size%sizeof(T)==0, "invalid binary size");
    std::vector<T> values(size/sizeof(T)); std::ifstream in(path,std::ios::binary);
    in.read(reinterpret_cast<char *>(values.data()),size);require(bool(in),"binary input read failed"); return values;
}
static std::vector<uint8_t> state(llama_context * ctx) {
    std::vector<uint8_t> bytes(llama_state_get_size(ctx));
    require(llama_state_get_data(ctx,bytes.data(),bytes.size())==bytes.size(),"incomplete state encoding");return bytes;
}
struct Material {
    int batch,length,outputs,chunk,threads,warmup,diagnostic,checkpoint;
    std::vector<llama_token> ids;
    explicit Material(const char * path) {
        std::ifstream in(path);in>>batch>>length>>outputs>>chunk>>threads>>warmup>>diagnostic>>checkpoint;
        require(batch>0 && batch<=512 && length>0 && length<=8192 && outputs>0 && outputs<=512 &&
                chunk>0 && chunk<=length && threads>0 && threads<=512 && warmup>=0 && warmup<=1,
                "invalid bounded material dimensions");
        require(!diagnostic || (batch<=2 && length<=32 && outputs<=8),"diagnostic material too large");
        ids.resize(batch*length);for(auto & id:ids)in>>id;require(bool(in),"incomplete token material");
    }
};
struct Batch {
    llama_batch value;
    explicit Batch(int size):value(llama_batch_init(size,0,1)) {require(value.token,"batch allocation failed");}
    ~Batch(){llama_batch_free(value);}
    void clear(){value.n_tokens=0;}
    void add(llama_token token,int seq,int pos,bool logits){
        int i=value.n_tokens++;value.token[i]=token;value.pos[i]=pos;value.n_seq_id[i]=1;
        value.seq_id[i][0]=seq;value.logits[i]=logits;
    }
};
struct Result { std::vector<llama_token> tokens;double total,prefill,decode; };
static llama_token greedy(float * logits,int vocab){
    require(logits,"missing output logits");int best=0;
    for(int i=0;i<vocab;++i){require(std::isfinite(logits[i]),"nonfinite output logits");if(logits[i]>logits[best])best=i;}
    return best;
}
static Result generate(llama_context * ctx,const Material & m,int vocab,int outputs,const std::string & diagnostic){
    auto begin=Clock::now();Batch batch(m.batch*m.chunk);std::vector<llama_token> next(m.batch),tokens(m.batch*outputs);
    std::ofstream logits_file;
    if(!diagnostic.empty())logits_file.open(diagnostic,std::ios::binary);
    auto collect=[&](const std::vector<int> & indices,bool record){
        for(int b=0;b<m.batch;++b){float * values=llama_get_logits_ith(ctx,indices[b]);next[b]=greedy(values,vocab);
            if(record && logits_file.is_open())logits_file.write(reinterpret_cast<char*>(values),vocab*sizeof(float));}
    };
    for(int start=0;start<m.length;start+=m.chunk){
        int end=std::min(m.length,start+m.chunk);batch.clear();std::vector<int> indices;
        for(int b=0;b<m.batch;++b){
            for(int t=start;t<end;++t)batch.add(m.ids[b*m.length+t],b,t,t==end-1);
            indices.push_back(batch.value.n_tokens-1);
        }
        require(llama_decode(ctx,batch.value)==0,"prefill decode rejected");
        collect(indices,end==m.length);
    }
    for(int b=0;b<m.batch;++b)tokens[b*outputs]=next[b];
    llama_synchronize(ctx);double prefill=seconds(begin);auto decoding=Clock::now();
    for(int step=1;step<outputs;++step){
        batch.clear();std::vector<int> indices;
        for(int b=0;b<m.batch;++b){batch.add(next[b],b,m.length+step-1,true);indices.push_back(b);}
        require(llama_decode(ctx,batch.value)==0,"token decode rejected");collect(indices,true);
        for(int b=0;b<m.batch;++b)tokens[b*outputs+step]=next[b];
    }
    llama_synchronize(ctx);
    if(logits_file.is_open()){logits_file.close();require(bool(logits_file),"logit write failed");}
    return {std::move(tokens),seconds(begin),prefill,seconds(decoding)};
}
static void next_state(llama_context * ctx,const Material & m,int vocab,const std::vector<llama_token> & tokens,const std::string & prefix){
    require(tokens.size()==size_t(m.batch*m.outputs),"wrong saved token count");Batch batch(m.batch);
    for(int b=0;b<m.batch;++b)batch.add(tokens[b*m.outputs+m.outputs-1],b,m.length+m.outputs-1,true);
    require(llama_decode(ctx,batch.value)==0,"continuation rejected");std::vector<float> logits;
    for(int b=0;b<m.batch;++b){float * row=llama_get_logits_ith(ctx,b);greedy(row,vocab);logits.insert(logits.end(),row,row+vocab);}
    save(prefix+"-logits.bin",logits);save(prefix+"-state.bin",state(ctx));
}
int main(int argc,char ** argv){
    try{
        require(argc==5,"usage: tide-lm-llama MODEL.gguf INPUT.txt OUTPUT_DIR run|restore");
        Material m(argv[2]);std::string out=argv[3];bool restore=std::string(argv[4])=="restore";
        require(restore || std::string(argv[4])=="run","unknown action");
        auto construction=Clock::now();llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
        auto model=llama_model_load_from_file(argv[1],mp);require(model,"model load failed");
        int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
        for(auto id:m.ids)require(id>=0 && id<vocab,"input token outside vocabulary");
        auto cp=llama_context_default_params();cp.n_seq_max=m.batch;cp.n_ctx=(m.length+m.outputs+32)*m.batch;
        cp.n_batch=m.batch*m.chunk;cp.n_ubatch=cp.n_batch;cp.n_threads=m.threads;cp.n_threads_batch=m.threads;
        cp.type_k=GGML_TYPE_BF16;cp.type_v=GGML_TYPE_BF16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv=false;cp.op_offload=false;cp.kv_unified=false;cp.no_perf=false;
        auto ctx=llama_init_from_model(model,cp);require(ctx,"context creation rejected (BF16 KV requested; no fallback)");
        double built=seconds(construction);
        if(restore){
            auto bytes=load<uint8_t>(out+"/state.bin");
            require(llama_state_set_data(ctx,bytes.data(),bytes.size())==bytes.size(),"incomplete state restoration");
            next_state(ctx,m,vocab,load<llama_token>(out+"/tokens.bin"),out+"/restored");
        }else{
            double warm=0;
            if(m.warmup){warm=generate(ctx,m,vocab,std::min(8,m.outputs),"").total;llama_memory_clear(llama_get_memory(ctx),true);}
            auto r=generate(ctx,m,vocab,m.outputs,m.diagnostic?out+"/logits.bin":"");save(out+"/tokens.bin",r.tokens);
            double checkpoint=0;
            if(m.checkpoint){auto start=Clock::now();save(out+"/state.bin",state(ctx));
                next_state(ctx,m,vocab,r.tokens,out+"/expected");checkpoint=seconds(start);}
            struct rusage usage{};getrusage(RUSAGE_SELF,&usage);
            std::ofstream metrics(out+"/metrics.json");metrics<<std::setprecision(17)
                <<"{\"total_seconds\":"<<r.total<<",\"prefill_seconds\":"<<r.prefill<<",\"decode_seconds\":"<<r.decode
                <<",\"construction_seconds\":"<<built<<",\"warmup_seconds\":"<<warm<<",\"checkpoint_seconds\":"<<checkpoint
                <<",\"host_peak_rss_bytes\":"<<usage.ru_maxrss*1024LL<<",\"vocab\":"<<vocab
                <<",\"n_batch\":"<<llama_n_batch(ctx)<<",\"n_ubatch\":"<<llama_n_ubatch(ctx)<<",\"n_ctx\":"<<llama_n_ctx(ctx)<<"}\n";
            metrics.close();require(bool(metrics),"metrics write failed");
        }
        llama_perf_context_print(ctx);llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
    }catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
}
