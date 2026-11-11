#pragma once
#include "weighted_column_layout.h"
namespace tide::weighted::columns {
// Finite module dispatch owns the typed private alternatives. The scheduler
// depends only on message/descriptor columns, not their internal state shapes.
LocalState initial_columns(const NodeWeights&,const MemoryConfiguration&,Index samples);
void validate_columns(const NodeWeights&,const MemoryConfiguration&,const LocalState&);
LocalWindow prepare_columns(const NodeWeights&,const MemoryConfiguration&,const Layout&,const Tensor&,const Tensor&,const LocalState&);
TensorFields read_columns(const NodeWeights&,const NodeBinding&,const LocalWindow&,const std::string&,const LocalState&);
Message full_columns(const NodeWeights&,const MemoryConfiguration&,const LocalWindow&,const Tensor&);
LocalState terminal_columns(const LocalWindow&);
long double workspace_bytes(const MemoryConfiguration&,const Layout&,const LocalState&,Index element_size);
}
