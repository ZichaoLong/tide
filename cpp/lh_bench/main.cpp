// Original C++ LH only. No edits to reference sources or Python execution.
#include "CortexNet.h"
#include "../bench/metrics_jsonl_writer.h"
#include <ATen/Parallel.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/resource.h>

namespace {
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
namespace AL = AccumulateLocal;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now()-start).count();
}
double rss() { rusage r{}; getrusage(RUSAGE_SELF, &r); return double(r.ru_maxrss)*1024; }
struct CountingSelector final : BaseSelector {
  NaiveSelector original;
  int64_t candidates=0, selected=0, node_jobs=0;
  CountingSelector(int64_t batch, int64_t budget, const std::shared_ptr<GraphConfig>& cfg, bool lead)
    : BaseSelector(batch, budget, cfg), original(batch, budget, cfg, lead, true) {}
  VPtrBatchSignals select(const VPtrBatchSignals& input) override {
    for (const auto& p : input) if (p) candidates += p->x.size(0);
    auto result = original.select(input);
    for (const auto& p : result) if (p) { selected += p->x.size(0); ++node_jobs; }
    return result;
  }
};
Json read(const std::string& path) { std::ifstream f(path); if (!f) throw std::runtime_error("cannot read "+path); return Json::parse(f); }
}

int main(int argc, char** argv) {
  try {
    std::string request, device, dtype;
    for (int i=1; i<argc; i+=2) {
      if (i+1>=argc) throw std::invalid_argument("missing CLI value");
      std::string flag=argv[i];
      if (flag=="--request") request=argv[i+1];
      else if (flag=="--device") device=argv[i+1];
      else if (flag=="--dtype") dtype=argv[i+1];
      else throw std::invalid_argument("unknown option "+flag);
    }
    if (request.empty() || device!="cpu" || (dtype!="float32" && dtype!="float64"))
      throw std::invalid_argument("--request PATH --device cpu --dtype float32|float64 required");
    auto r=read(request), input=read(r.at("input"));
    const int64_t batch=r.at("batch"), steps=r.at("steps"), warmup=r.at("warmup"), reps=r.at("repetitions"), threads=r.at("threads");
    const std::string mode=r.at("mode"), out=r.at("output_dir");
    if (batch<1 || batch>512 || steps<1 || steps>256 || warmup<0 || warmup>64 || reps<1 || reps>20 || threads<1 || threads>56
        || (mode!="nograd" && mode!="grad-forward" && mode!="backward"))
      throw std::invalid_argument("request exceeds bounded LH benchmark domain");
    if (!std::filesystem::create_directory(out)) throw std::runtime_error("native output directory must be new");
    at::set_num_threads(threads); at::set_num_interop_threads(1); torch::manual_seed(r.at("seed").get<int64_t>());
    AL::TensorHidden::is_no_grad=AL::KVHidden::is_no_grad=(mode=="nograd");
    AL::TensorHidden::multi_batch_forward_mode=AL::KVHidden::multi_batch_forward_mode=true;
    AL::KVHidden::attention_mode=AL::KVHidden::AttentionMode::CROSSBATCH;
    at::AutoGradMode grad(mode!="nograd");
    auto started=Clock::now(), construct=Clock::now();
    auto cfg=std::make_shared<GraphConfig>(input.at("graph_dir").get<std::string>()+"/cfg.json");
    GraphData gd(input.at("graph_dir").get<std::string>());
    auto model_cfg=input.at("model");
    const int64_t width=r.value("width", int64_t(model_cfg["input_"]["emitD"]));
    if (width<4 || width>2048 || width%4) throw std::invalid_argument("width must be a multiple of four in [4,2048]");
    const int64_t edges=gd.inputA.nnz+gd.outputA.nnz+gd.ioA.nnz+gd.oiA.nnz;
    const int64_t estimated=edges*width*width+(2*cfg->levelptr.back()+2*model_cfg.at("vocab_size").get<int64_t>())*width
      +edges+1+model_cfg.at("n_layer").get<int64_t>();
    if (estimated>12000000000LL) throw std::invalid_argument("reconstruction exceeds 12B parameter bound");
    for (auto name : {"input_", "output_", "iobridge_", "oibridge_"}) model_cfg[name]["emitD"]=model_cfg[name]["receiveD"]=width;
    IOCortexNet model(model_cfg, cfg, gd.inputA, gd.outputA, gd.ioA, gd.oiA);
    model.to(dtype=="float32" ? torch::kFloat32 : torch::kFloat64); model.eval();
    int64_t params=0; for (const auto& p : model.parameters()) params+=p.numel();
    if (params!=estimated || (width==int64_t(input["model"]["input_"]["emitD"])
        && params!=input.at("expected_parameters").get<int64_t>()))
      throw std::runtime_error("original model parameter count differs from preflight");
    const double construction=elapsed(construct);
    Json model_record{{"parameters",params},{"static_nodes",cfg->levelptr.back()},{"width",width},{"batch",batch},
      {"dtype",dtype},{"mode",mode},{"threads",threads},{"construction_seconds",construction},{"model_config",model_cfg},
      {"vocab",model.config.vocab_size},{"layers",model.config.n_layer},{"selectnum",input.at("selectnum")},
      {"peak_rss_bytes_after_construction",rss()},{"autograd_retention","continuous within a repetition; reset between repetitions"}};
    { std::ofstream f(out+"/model.json"); f << model_record.dump(2) << '\n'; }
    std::cout << model_record.dump() << '\n' << std::flush;
    portable_experiment::MetricsJsonlWriter metrics(out+"/metrics.jsonl",r.at("run_id"));
    int64_t event=0;
    auto ids=at::randint(0,model.config.vocab_size,{warmup+steps,batch},at::kLong);
    for (int64_t rep=0; rep<reps; ++rep) {
      CountingSelector is(batch,input.at("selectnum"),cfg,input.at("with_lead_point")), os(batch,input.at("selectnum"),cfg,input.at("with_lead_point"));
      VPtrBatchSignals ia,oa; AL::VPtrBatchPtrBaseHidden ih,oh; AL::PtrBatchPtrBaseHidden ph;
      at::Tensor loss;
      for (int64_t t=0; t<warmup+steps; ++t) {
        auto c0=is.candidates+os.candidates, a0=is.selected+os.selected, j0=is.node_jobs+os.node_jobs;
        auto tic=Clock::now();
        auto logits=model.think(ids[t],is,os,ia,oa,ih,oh,ph);
        const double seconds=elapsed(tic);
        if (logits.requires_grad() != (mode!="nograd")) throw std::runtime_error("unexpected output autograd mode");
        if (!at::isfinite(logits).all().item<bool>()) throw std::runtime_error("nonfinite logits");
        if (mode=="backward") loss=loss.defined() ? loss+logits.square().mean() : logits.square().mean();
        if (t<warmup) continue;
        auto count=is.candidates+os.candidates-c0, active=is.selected+os.selected-a0, jobs=is.node_jobs+os.node_jobs-j0;
        metrics.Write(event++,{{"perf/forward_seconds",seconds},{"perf/ms_per_sample_token",seconds*1000/batch},
          {"perf/tokens_per_second",batch/seconds},{"memory/process_peak_rss_bytes",rss()},
          {"work/candidates",double(count)},{"work/selected",double(active)},{"work/selected_node_jobs",double(jobs)},
          {"work/selected_fraction",double(active)/(batch*2*cfg->levelptr.back()*model.config.n_layer)},
          {"check/logit_sum",logits.detach().to(at::kDouble).sum().item<double>()}},elapsed(started),
          {{"repetition",rep},{"token_step",t},{"phase",std::string("forward")}});
        std::cout << "rep=" << rep << " token=" << t << " forward_seconds=" << seconds << " selected=" << active << '\n' << std::flush;
      }
      if (mode=="backward") {
        auto tic=Clock::now(); loss.backward(); const double seconds=elapsed(tic);
        metrics.Write(event++,{{"perf/backward_seconds",seconds},{"memory/process_peak_rss_bytes",rss()}},elapsed(started),
                      {{"repetition",rep},{"phase",std::string("backward")}});
        model.zero_grad();
      }
    }
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
