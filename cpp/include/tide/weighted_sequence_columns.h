#pragma once
// 0770593 architecture/KV/Settle: typed tensor windows, physical message support.
// Host metadata is int64. Floating columns are read-only autograd-owned values.
#include "tide/weighted_columns.h"
#include "tide/settle.h"

namespace tide::weighted::columns {
struct Layout {
  std::vector<Index> offsets, times;
  Index samples() const { return static_cast<Index>(offsets.size())-1; }
  void validate() const;
  Layout select(const std::vector<Index>& rows) const;
};
struct KVState {
  Layout layout;
  TensorFields fields; // keys, values, summaries, weights; no floating timestamps.
  Tensor inert;
  std::vector<Index> last_times, observations;
  static KVState from_states(const std::vector<State>&);
  std::vector<State> states() const;
  void validate(const NodeWeights&,const KVConfig&) const;
};
struct KVWindow {
  Layout layout, bank_layout;
  Tensor mass, content;
  TensorFields bank;
  KVState terminal;
};
struct RecurrentState {
  TensorFields fields; // H: B,M,dv,dk; Z: B,2,ds; W: B,2.
  Tensor inert;
  std::vector<Index> last_times,observations;
  Layout cache_layout; // Empty offsets for basic affine/delta, exact slots otherwise.
  Tensor cache_values;
  static RecurrentState from_states(const std::vector<State>&);
  std::vector<State> states() const;
  void validate(const NodeWeights&,const RecurrentConfig&) const;
};
struct RecurrentWindow {
  Layout layout,write_layout;
  Tensor mass,content;
  TensorFields bank;
  std::vector<Index> times;
  Layout cache_layout;
  Tensor cache_values;
  RecurrentState terminal;
};
using LocalState=std::variant<KVState,RecurrentState>;
using LocalWindow=std::variant<KVWindow,RecurrentWindow>;
struct Message {
  std::vector<Index> rows;
  Tensor strength,content;
};
struct NodeResult {
  std::vector<Index> rows;
  LocalWindow window;
  TensorFields descriptors;
  Tensor controls;
};
struct Result {
  Layout layout;
  std::map<Index,LocalState> states;
  std::map<Index,Message> outputs,messages;
  std::map<Index,NodeResult> nodes;
  std::map<std::string,Index> stats;
};
bool supported(const KVConfig&);
Tensor logit_attention(const Tensor& logits,const Tensor& values,const Tensor& weights);
KVWindow prepare(const NodeWeights&,const KVConfig&,const Layout&,const Tensor& mass,
                 const Tensor& content,const KVState&);
TensorFields read(const NodeWeights&,const KVConfig&,const RegionalConfig&,const KVWindow&,
                  const std::string& mode,const KVState& initial);
Message full(const NodeWeights&,const KVConfig&,const KVWindow&,const Tensor& controls);
RecurrentWindow prepare(const NodeWeights&,const RecurrentConfig&,const Layout&,const Tensor& mass,
                        const Tensor& content,const RecurrentState&);
TensorFields read(const NodeWeights&,const RecurrentConfig&,const RegionalConfig&,const RecurrentWindow&,
                  const std::string& mode,const RecurrentState& initial);
Message full(const NodeWeights&,const RecurrentConfig&,const RecurrentWindow&,const Tensor& controls);
// Independent C++ region-order schedule; no Python or generic-engine callbacks.
Result settle(const SettleGraph&,const Model&,const Layout&,const std::map<Index,Message>& ports,
              const std::map<Index,LocalState>& left={},Index max_workspace_bytes=Index{4}*1024*1024*1024);
Tensor readout(const Result&,const Tensor& like);
}
