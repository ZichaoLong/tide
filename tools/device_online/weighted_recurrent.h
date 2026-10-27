#pragma once
// Fixed recurrent textbooks §1–3: actual Upd, Mem and unscaled Full.
#include "weighted_row_graph.h"
namespace tide::device_online {
using WeightedRecurrentParameters=std::map<std::string,at::Tensor>;
enum class WeightedRecurrentAction {Update,Memory,Full};
std::map<std::string,std::vector<int64_t>> weighted_recurrent_parameter_shapes(int64_t banks,const weighted::RecurrentConfig&);
struct WeightedRecurrentSnapshot {
  WeightedRawLayout layout;
  WeightedRawState state;
  at::Tensor owners,decay,present;
};
WeightedRecurrentSnapshot append_weighted_recurrent_snapshot(DeviceProgram&,const weighted::RecurrentConfig&,
  const WeightedRawState&,const at::Tensor& time,const at::Tensor& mass,const at::Tensor& owners,const at::Tensor& present,
  int64_t banks,bool positive_only,const at::Tensor& error,int64_t budget);
struct WeightedRecurrentTape {
  weighted::RecurrentConfig config;WeightedRecurrentAction action;
  WeightedRecurrentSnapshot snapshot;
  std::shared_ptr<WeightedRowGraph> graph;
  std::map<std::string,size_t> inputs,outputs;
  std::map<std::string,at::Tensor> fields;
  std::optional<WeightedRawUpdateTape> update;
  WeightedRecurrentParameters parameters;
};
// Memory reads explicitly supplied initial/old/proposal raw state. Full caller
// supplies proposal + actual active mask. Update retains every positive receive.
WeightedRecurrentTape append_weighted_recurrent(DeviceProgram&,const weighted::RecurrentConfig&,WeightedRecurrentAction,
  const WeightedRawState&,const at::Tensor& time,const at::Tensor& mass,const at::Tensor& content,
  const at::Tensor& owners,const at::Tensor& present,const WeightedRecurrentParameters&,
  const at::Tensor& error,int64_t budget=256*1024*1024);
struct WeightedRecurrentGradient {
  at::Tensor raw,raw_connected;
  WeightedRowAdjoint mass,content;
  std::map<std::string,WeightedRowAdjoint> parameters;
};
// For Update, fields are raw named floating fields, with padding retaining no
// semantic value; Memory has four independent roots; Full has only G.
WeightedRecurrentGradient append_weighted_recurrent_vjp(DeviceProgram&,const WeightedRecurrentTape&,
  const std::map<std::string,WeightedRowAdjoint>& roots,const at::Tensor& error,int64_t budget=256*1024*1024);
}
