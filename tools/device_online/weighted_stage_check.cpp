#include "weighted_stage_check.h"
#include "portable_torch/runtime.hpp"
#include <ATen/Parallel.h>
#include <iostream>
namespace tide::device_online::stage_check {
void run(at::Device device,const std::string& family,const std::string& read){
  auto f=fixture(family,read);at::NoGradGuard ng;auto options=at::TensorOptions().dtype(at::kFloat).device(device);
  auto error=at::zeros({1},options.dtype(at::kInt));DeviceProgram p(device);p.limit_workspace(128*1024*1024);
  TaskServices services(p,device,128*1024*1024);WeightedFlowNodes nodes;
  const int64_t budget=768*1024*1024;
  auto remaining=[&](DeviceProgram& current){const auto used=current.retained_tensor_bytes({},services.programs());
    if(used>=budget)throw std::invalid_argument("weighted stage total tensor budget exceeded");return budget-used;};
  for(int64_t v=0;v<2;++v){nodes.push_back(std::make_unique<WeightedFlowNode>(f.model,v,f.initial,device,4,budget));auto& n=*nodes.back();
    auto& worker=services.begin();n.source=bind_weighted_flow_source(worker,n.banks,f.graph,v,n.binding,error,remaining(worker));services.end(worker);}
  DeviceReady planner({0,0},1,{},rows,device,true,{1});auto ready=planner.append_stage(p,atoms(f,device),at::full({1},12,options.dtype(at::kLong)),error);
  for(int64_t v=0;v<2;++v){auto& n=*nodes[v];n.input=append_weighted_node_input(p,ready,n.source.classes,v,rows,f.graph.inputs.size(),error);
    append_weighted_node_local(p,services,n,read,error,remaining);}
  WeightedFlowRegion region(f.graph,f.model,0,f.initial,device);append_weighted_region_local(p,services,region,nodes,error,remaining);
  for(auto& n:nodes)append_weighted_node_full(services,*n,error,remaining);
  std::vector<WeightedNodeRoots> output_roots;std::map<int64_t,WeightedRowAdjoint> control_roots;
  for(int64_t v=0;v<2;++v){output_roots.push_back(roots(*nodes[v]));control_roots[v]={at::zeros({rows,1},options),at::zeros({rows},options.dtype(at::kBool))};}
  auto region_gradient=append_weighted_flow_region_vjp(p,services,region,nodes,control_roots,error,remaining);
  std::vector<WeightedNodeGradient> gradients;auto partials=region_gradient.parameters;
  for(int64_t v=0;v<2;++v){auto root=output_roots[v];
    for(auto& [name,part]:root.describe){auto combined=weighted_reverse::zeros(p,part.value);
      weighted_reverse::add(p,combined,part,nodes[v]->input.bank,error);
      weighted_reverse::add(p,combined,region_gradient.describe.at(v).at(name),nodes[v]->input.bank,error);part=combined;}
    gradients.push_back(append_weighted_node_vjp(p,services,*nodes[v],read,root,error,remaining));
    partials.insert(partials.end(),gradients.back().parameters.begin(),gradients.back().parameters.end());}
  auto registry=f.model.parameters();auto parameters=append_weighted_parameter_vjp(p,registry,partials,error,remaining(p));
  services.append_stop();services.finish();p.finish();
  auto execute=[&](){portable_torch::synchronize(device);services.submit();p.run(60000);services.wait(60000);
    const auto code=error.cpu().item<int>();if(code)throw std::runtime_error("weighted stage device error="+std::to_string(code));};
  set_roots(nodes,output_roots,control_roots,"none");execute();
  Result reference;{at::AutoGradMode grad(true);Options o;o.packed=false;reference=Greedy(f.graph,f.model,o).run(f.initial,f.inputs,12,12);}
  std::vector<std::string> tests{"none","content","control","full","all"};
  for(const auto& [name,_]:nodes[0]->describe->fields)tests.push_back("describe."+name);
  for(const auto& field:nodes[0]->layout.fields())tests.push_back("raw."+field.name);
  for(const auto& test:tests){set_roots(nodes,output_roots,control_roots,test);execute();
    try{verify(f,reference,ready,nodes,output_roots,control_roots,gradients,parameters,test);}catch(...){std::cerr<<"root="<<test<<'\n';throw;}}
  set_roots(nodes,output_roots,control_roots,"content");output_roots[0].content.value[0][0]=NAN;
  bool refused=false;try{execute();}catch(const std::runtime_error& e){refused=std::string(e.what())=="weighted stage device error=37";}
  if(!refused)throw std::runtime_error("connected nonfinite stage root accepted");
  p.close();services.close();std::cout<<"weighted-stage case "<<family<<' '<<read<<" passed roots="<<tests.size()<<'\n'<<std::flush;
}
}
int main(int argc,char** argv){portable_torch::RuntimeSession runtime;
  try{auto args=portable_torch::parse_cli(argc,argv,true);if(args.help){portable_torch::print_usage(std::cout,argv[0]);return 0;}
    if(args.device_spec=="auto"||args.dtype!=at::kFloat)throw std::invalid_argument("weighted stage requires explicit accelerator FP32");
    auto device=portable_torch::resolve_device(args);at::set_num_threads(1);at::set_num_interop_threads(1);
    for(const std::string family:{"kv","affine","delta","gated_delta","ssd","kv_query","kv_node_gain","kv_attention_gain"})for(const std::string read:{"content","old","proposal"}){
      try{tide::device_online::stage_check::run(device,family,read);}catch(...){std::cerr<<family<<' '<<read<<'\n';throw;}}
    std::cout<<"weighted-stage: passed configurations=24 independent_roots=true canonical_parameters=true raw_fields=true connected_nonfinite_refusal=true\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
