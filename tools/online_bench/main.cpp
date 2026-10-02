#include "consumer.h"
#include "failure.h"
#include <ATen/Parallel.h>
#include <filesystem>
#include <iostream>

int main(int argc,char** argv) {
  portable_torch::RuntimeSession runtime;std::string output;
  try {
    auto c=tide_flow::parse(argc,argv);
    if(c.runtime.help){portable_torch::print_usage(std::cout,argv[0]);
      std::cout<<"--packet PATH --family pdg|timed-dag|settle --preset cpu|mixed-a|mixed-b|mixed-c|resident --schedule streaming|prefill\n"
        "--training --optimizer sgd|adamw --steps N --warmup N --windows-per-step N --threads N --parameter-budget BYTES --diagnostics\n"
        "--workers N --packed-sources --batch-next (eager native only; threads sets ATen intra-op parallelism)\n"
        "--sample-chunk-rows N (physical sample maximum; 0 keeps the whole logical batch)\n"
        "--devices N --owner-policy memory|locality --chunk-policy conservative|aggressive\n"
        "--resident-context-bytes BYTES (per-device saved-state pool; positive enables compact rows, 0 retains dense storage)\n"
        "--device-memory-bytes BYTES (resident incremental per-device cap; 0 uses driver free memory) --head-workspace-bytes BYTES\n"
        "--resident-{queue,arrivals,outputs,trace,stages,workspace-bytes,full-chunk-rows,emission-chunk-rows,aggregate-chunk-rows,attention-chunk-rows,attention-key-rows,kv-rows,kv-trace-rows,max-repeat-ticks,retained-bytes,backward-bytes,optimizer-bytes,program-workspace-bytes,reverse-chunk-rows} N\n";return 0;}
    auto p=tide_flow::read_packet(c.packet);auto device=portable_torch::resolve_device(c.runtime);
    at::set_num_threads(c.threads);at::set_num_interop_threads(1);
    if(!std::filesystem::create_directory(c.runtime.output_dir))throw std::invalid_argument("output directory must be new");
    output=c.runtime.output_dir;std::ostringstream diagnostics;
    auto result=tide_flow::run(p,c,device,c.diagnostics?&diagnostics:nullptr);
    runtime.close();
    if(c.diagnostics)tide_flow::atomic_text(output+"/diagnostics.jsonl",diagnostics.str());
    tide_flow::atomic_text(output+"/result.json",result);std::cout<<result;return 0;
  }catch(const tide_flow::RecordedFailure& error){
    if(!output.empty())try{tide_flow::atomic_text(output+"/result.json",error.record);}catch(...){}
    std::cerr<<error.what()<<'\n';return 2;
  }catch(const std::exception& error){
    if(!output.empty())try{tide_flow::atomic_text(output+"/result.json","{\"state\":\"failed\",\"error\":"+tide_flow::quoted(error.what())+"}\n");}catch(...){}
    std::cerr<<error.what()<<'\n';return 2;
  }
}
