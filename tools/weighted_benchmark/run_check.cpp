// CLI and durable external-resume metadata checks; graph payload codecs have
// their own public C++ qualification, and real end-to-end resume is separate.
#include "cli.h"
#include "run_checkpoint.h"
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace weighted_benchmark;
namespace {
void require(bool value,const char* what){if(!value)throw std::runtime_error(what);}
template<class F> void rejects(F test){try{test();}catch(const std::invalid_argument&){return;}throw std::runtime_error("invalid request accepted");}
Request parse(std::vector<std::string> arguments) {
  arguments.insert(arguments.begin(),"weighted-benchmark");std::vector<char*> argv;
  for(auto& value:arguments)argv.push_back(value.data());return parse_request(argv.size(),argv.data());
}
class CheckpointOwner : public Resident {
 public:
  Index value=30;
  Step step(Index)override{throw std::logic_error("not a scheduler test");}
  std::map<std::string,Tensor> parameters()const override{return {};}
  TrainingState training_state()const override{return {};}
  Continuation checkpoint()const override{Continuation q;q.cut=value;return q;}
  Index cut()const override{return value;}
  void save(const std::filesystem::path& path,int64_t)const override{std::ofstream f(path);f<<value;f.close();}
  void restore(const std::filesystem::path& path,int64_t)override{std::ifstream f(path);if(!(f>>value))throw std::invalid_argument("invalid test payload");}
  void close()override{}
};
}
int main() {
  const auto root=std::filesystem::temp_directory_path()/("tide-run-check-"+std::to_string(getpid()));
  try {
    const std::vector<std::string> base{"--fixture","fixture.json","--device","cpu","--output-dir","new"};
    auto request=parse(base);require(request.validation=="strict"&&request.return_mode=="complete","strict defaults changed");
    auto rooted=base;rooted.push_back("--check-roots");require(parse(rooted).check_roots,"bare independent-root flag ignored");
    rooted.push_back("false");require(!parse(rooted).check_roots,"explicit false independent-root flag ignored");
    rejects([&]{auto a=base;a.insert(a.end(),{"--check-roots","--validation","none"});parse(a);});
    auto selected=base;selected.insert(selected.end(),{"--validation","none","--return-mode","metrics","--threads","3","--communication","cpu_bridge","--warmup","0","--steps","9"});
    request=parse(selected);require(request.threads==3&&request.return_mode=="metrics"&&request.warmup==0&&request.steps==9,"public run options lost");
    require(request.input_interface=="auto","default tensor input policy missing");
    auto recovered=selected;recovered.insert(recovered.end(),{"--checkpoint-check","true"});
    require(parse(recovered).checkpoint,"selected-return exact raw resume was rejected");
    rejects([&]{auto a=base;a.insert(a.end(),{"--return-mode","outputs"});parse(a);});
    rejects([&]{auto a=base;a.insert(a.end(),{"--steps","2garbage"});parse(a);});
    rejects([&]{auto a=base;a.insert(a.end(),{"--loss-scale","nan"});parse(a);});
    rejects([&]{auto a=base;a.insert(a.end(),{"--input-interface","invented"});parse(a);});
    rejects([&]{auto a=base;a.insert(a.end(),{"--resident-executor","block-v2"});parse(a);});
    auto block=base;block.insert(block.end(),{"--device","cuda:0","--implementation","resident","--return-mode","outputs","--validation","none"});
    require(parse(block).resident_executor=="auto","resident auto default missing");
    auto explicit_block=block;explicit_block.insert(explicit_block.end(),{"--resident-executor","block-v2","--resident-block-limits","block.json"});
    require(parse(explicit_block).block_limits.is_absolute(),"block budget path not resolved");
    rejects([&]{auto a=explicit_block;a.insert(a.end(),{"--resident-limits","general.json"});parse(a);});
    rejects([&]{auto a=explicit_block;a.insert(a.end(),{"--return-mode","complete"});parse(a);});
    rejects([&]{auto a=explicit_block;a.insert(a.end(),{"--resident-executor","general"});parse(a);});
    rejects([&]{auto a=base;a.insert(a.end(),{"--node-devices","cuda:0,"});parse(a);});
    Json limits={{"queue",64},{"prefill",true},{"retained_bytes",100}};
    apply_resident_limits(limits,{{"queue",128},{"prefill",true}},false,true);
    require(limits.at("queue")==128,"resident integer override lost");
    rejects([&]{apply_resident_limits(limits,{{"prefill",false}},false,true);});
    rejects([&]{apply_resident_limits(limits,{{"queue",true}},false,true);});
    rejects([&]{apply_resident_limits(limits,{{"retained_bytes",200}},false,true);});
    rejects([&]{apply_resident_limits(limits,{{"unknown",3}},true,true);});
    require(limits.at("queue")==128&&limits.at("retained_bytes")==100,"rejected limits mutated target");
    if(!std::filesystem::create_directory(root))throw std::runtime_error("test directory already exists");
    CheckpointOwner owner;
    const Json fixture={{"fixture_sha256","one"},{"span",10}};
    const auto identity=request.checkpoint_identity(fixture);
    auto block_request=parse(block);auto block_identity=block_request.checkpoint_identity(fixture);
    require(block_identity.at("resident_executor")=="auto","resident policy absent from identity");
    block_request.resident_executor="general";
    require(!block_request.checkpoint_identity(fixture).contains("resident_executor"),"explicit general changed historical identity");
    auto effective=block_identity;effective["resident_execution"]={{"effective","block-v2"},{"checkpoint_codec","block-schema-2"}};
    save_run_checkpoint(root/"block-checkpoint",owner,effective,3,1000);
    checkpoint_manifest(root/"block-checkpoint",block_identity,true);
    rejects([&]{checkpoint_manifest(root/"block-checkpoint",block_identity);});
    rejects([&]{auto changed=effective;changed["resident_execution"]["effective"]="general";restore_run_checkpoint(root/"block-checkpoint",owner,changed,1000);});
    require(restore_run_checkpoint(root/"block-checkpoint",owner,effective,1000)==3,"effective block identity did not restore");
    require(identity.at("input_policy")=="tensor-window-v1; compatible Settle modules, explicit graph fallback",
            "tensor window identity missing");
    require(identity.at("position_policy")=="bounded-common-origin-v1","column position policy missing");
    auto records=request;records.input_interface="records";
    require(!records.checkpoint_identity(fixture).contains("input_policy"),"explicit records silently adopt column policy");
    require(!records.checkpoint_identity(fixture).contains("position_policy"),"explicit records silently adopt new position algorithm");
    auto complete=request;complete.return_mode="complete";
    require(!complete.checkpoint_identity(fixture).contains("position_policy"),"complete records changed historical position identity");
    require(identity.at("record_policy")=="selected-boundary-v1","selected-return policy identity missing");
    auto historical=request;historical.trace_records="complete";
    require(!historical.checkpoint_identity(fixture).contains("record_policy"),"historical trace identity changed");
    require(identity.at("query_execution")=="auto"&&identity.at("query_policy")=="kv-block-v1+packed-v2+describe-batch-v2","query policy missing from checkpoint");
    require(identity.at("scheduler")=="component-window","prefill policy missing from checkpoint");
    auto greedy=request;greedy.prefill_schedule="greedy";
    require(!greedy.checkpoint_identity(fixture).contains("prefill_schedule"),"legacy scheduler identity changed");
    save_run_checkpoint(root/"checkpoint",owner,identity,3,1000);owner.value=0;
    require(restore_run_checkpoint(root/"checkpoint",owner,identity,1000)==3&&owner.value==30,"external resume cut/index mismatch");
    rejects([&]{auto other=identity;other["fixture_sha256"]="two";checkpoint_manifest(root/"checkpoint",other);});
    rejects([&]{auto other=identity;other["query_policy"]="kv-packed-v2+describe-batch-v1";checkpoint_manifest(root/"checkpoint",other);});
    rejects([&]{auto other=identity;other.erase("position_policy");checkpoint_manifest(root/"checkpoint",other);});
    auto old=request;old.query_execution="legacy";
    require(!old.checkpoint_identity(fixture).contains("query_execution"),"legacy checkpoint identity changed");
    rejects([&]{checkpoint_manifest(root/"checkpoint",old.checkpoint_identity(fixture));});
    rejects([&]{checkpoint_manifest(root/"checkpoint",records.checkpoint_identity(fixture));});
    rejects([&]{checkpoint_manifest(root/"checkpoint",greedy.checkpoint_identity(fixture));});
    auto other=request;other.query_execution="packed-v2";
    rejects([&]{checkpoint_manifest(root/"checkpoint",other.checkpoint_identity(fixture));});
    rejects([&]{save_run_checkpoint(root/"checkpoint",owner,identity,3,1000);});
    owner.value=40;save_run_checkpoint(root/"checkpoint",owner,identity,4,1000);
    require(!std::filesystem::exists(root/"checkpoint/window-3")&&std::filesystem::exists(root/"checkpoint/window-4/manifest.json"),"checkpoint generations not rotated after publication");
    owner.value=0;require(restore_run_checkpoint(root/"checkpoint",owner,identity,1000)==4&&owner.value==40,"latest checkpoint did not resume");
    std::filesystem::remove_all(root);std::cout<<"public CLI and run-checkpoint checks passed\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';std::filesystem::remove_all(root);return 1;}
}
