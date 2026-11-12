#pragma once
#include "weighted_column_layout.h"
namespace tide::weighted::columns {
// Finite module dispatch owns the typed private alternatives. The scheduler
// depends only on message/descriptor columns, not their internal state shapes.
LocalState initial_columns(const NodeWeights&,const MemoryConfiguration&,Index samples);
void validate_columns(const NodeWeights&,const MemoryConfiguration&,const LocalState&);
LocalWindow prepare_columns(const NodeWeights&,const MemoryConfiguration&,const Layout&,const Tensor&,const Tensor&,const LocalState&);
struct LocalPrepareRequest {
  const NodeWeights* weights;
  MemoryConfiguration config;
  Layout layout;
  Tensor mass,content;
  LocalState left;
};
std::vector<LocalWindow> prepare_columns_many(const std::vector<LocalPrepareRequest>&,Index max_workspace_bytes);
TensorFields read_columns(const NodeWeights&,const NodeBinding&,const LocalWindow&,const std::string&,const LocalState&);
Message full_columns(const NodeWeights&,const MemoryConfiguration&,const LocalWindow&,const Tensor&);
struct LocalReadRequest {
  const NodeWeights* weights;
  NodeBinding binding;
  LocalWindow window;
  std::string mode;
  LocalState initial;
};
struct LocalFullRequest {
  const NodeWeights* weights;
  MemoryConfiguration config;
  LocalWindow window;
  Tensor controls;
};
std::vector<TensorFields> read_columns_many(const std::vector<LocalReadRequest>&,Index max_workspace_bytes);
std::vector<Message> full_columns_many(const std::vector<LocalFullRequest>&,Index max_workspace_bytes);
LocalState terminal_columns(const LocalWindow&);
long double workspace_bytes(const MemoryConfiguration&,const Layout&,const LocalState&,Index element_size);
}
