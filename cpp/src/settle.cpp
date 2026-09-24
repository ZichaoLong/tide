#include "tide/settle.h"
#include "tide/kernel.h"
#include "tide/ops.h"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace tide {
SettleGraph::SettleGraph(Graph body, std::vector<Index> ranks)
    : body_(std::move(body)), ranks_(std::move(ranks)) {
  body_.compile();
  if (ranks_.size() != body_.regions.size() ||
      std::set<Index>(ranks_.begin(), ranks_.end()).size() != ranks_.size() ||
      std::any_of(ranks_.begin(), ranks_.end(), [](Index r) {
        return r <= 0 || r > std::numeric_limits<Index>::max() - 2;
      })) throw std::invalid_argument("SettleGraph requires distinct positive representable region ranks");
  if (body_.inputs.empty() || body_.outputs.empty())
    throw std::invalid_argument("SettleGraph requires input and output boundaries");
  stride_ = *std::max_element(ranks_.begin(), ranks_.end()) + 2;
  for (const auto& edge : body_.edges)
    if (edge.delay != rank(edge.target) - rank(edge.source))
      throw std::invalid_argument("SettleGraph edges must follow region ranks with matching delays");
  encoded_ = body_;
  const Index n = body_.nodes.size(), r = body_.regions.size();
  encoded_.nodes.push_back(Node{r, false, true});
  encoded_.nodes.push_back(Node{r + 1, false, true});
  encoded_.regions.push_back(Region{1});
  encoded_.regions.push_back(Region{1});
  auto& layout = *encoded_.layout;
  auto& domain = *encoded_.source_domain;
  for (Index p = 0; p < static_cast<Index>(body_.inputs.size()); ++p) {
    const auto node = body_.inputs[p];
    const Index edge = encoded_.edges.size();
    encoded_.edges.push_back({n, node, rank(node)});
    layout.edge_source.push_back(p);
    layout.edge_target.push_back(body_.layout->input[p]);
    domain.edge_target.push_back(body_.source_domain->input[p]);
    encoded_.origins.push_back({edge, p, stride_});
  }
  for (Index p = 0; p < static_cast<Index>(body_.outputs.size()); ++p) {
    const auto node = body_.outputs[p];
    encoded_.edges.push_back({node, n + 1, output_rank() - rank(node)});
    layout.edge_source.push_back(body_.layout->output[p]);
    layout.edge_target.push_back(p);
    domain.edge_target.push_back(p);
  }
  encoded_.inputs = {n}; encoded_.outputs = {n + 1};
  layout.input = {0}; layout.output = {0}; domain.input = {0};
  encoded_.compile();
  encoded_.topological_order();
}

Index SettleGraph::rank(Index node) const { return ranks_.at(body_.nodes.at(node).region); }

Model SettleGraph::embed_model(const Model& body) const {
  Model result = body;
  configure_model(body_, result); validate_model(body_, result);
  const auto options = body.nodes[0].bias.options().requires_grad(false);
  for (int i = 0; i < 2; ++i) {
    result.nodes.push_back({at::zeros({body.width()}, options), at::zeros({body.width(), body.width()}, options),
                            at::zeros({body.width()}, options), at::zeros({body.width()}, options)});
    result.regions.push_back({});
  }
  auto one = [&] { return at::ones({}, options); };
  result.input_scale = {one()}; result.output_scale = {one()};
  result.agg_scale.insert(result.agg_scale.end(), body.input_scale.begin(), body.input_scale.end());
  result.agg_scale.insert(result.agg_scale.end(), body.output_scale.begin(), body.output_scale.end());
  for (size_t i = 0; i < body_.inputs.size() + body_.outputs.size(); ++i) result.edge_scale.push_back(one());
  configure_model(encoded_, result); validate_model(encoded_, result);
  return result;
}

Continuation SettleGraph::embed_initial(const Continuation& initial) const {
  if (initial.identity != body_.identity || initial.batch_size < 1 || initial.cut != 0 ||
      !initial.pending.empty() || !initial.ledger.empty())
    throw std::invalid_argument("SettleGraph embed_initial requires body identity and an initial cut");
  auto owners = [&](const auto& values, Index count) {
    for (const auto& item : values) {
      const auto [b, v] = item.first;
      if (b < 0 || b >= initial.batch_size || v < 0 || v >= count)
        throw std::invalid_argument("invalid body initial owner in SettleGraph embedding");
    }
  };
  owners(initial.states, body_.nodes.size()); owners(initial.history, body_.regions.size());
  auto result = initial; result.identity = encoded_.identity;
  return result;
}

std::vector<External> SettleGraph::external(const Tensor& values, Index start, bool encoded) const {
  if (!values.defined() || values.dim() != 3 || values.size(0) < 1 || values.size(2) < 1)
    throw std::invalid_argument("SettleGraph values require [batch, sequence, width]");
  // Bound the complete stop as well as every event, without intermediate overflow.
  if (start < 0 || start > std::numeric_limits<Index>::max() / stride_ ||
      values.size(1) > std::numeric_limits<Index>::max() / stride_ - start)
    throw std::invalid_argument("SettleGraph position clock overflow");
  std::vector<External> result;
  const Index ports = encoded ? 1 : body_.inputs.size();
  for (Index b = 0; b < values.size(0); ++b)
    for (Index p = 0; p < ports; ++p)
      for (Index t = 0; t < values.size(1); ++t) {
        const auto position = start + t;
        result.push_back({b, p, position, stride_ * position + (encoded ? 0 : rank(body_.inputs[p])), values[b][t]});
      }
  return result;
}

SettleExecutor::SettleExecutor(SettleGraph spec, Model model, Options options, std::string algorithm)
    : spec_(std::move(spec)) {
  if (algorithm == "frontier")
    frontier_ = std::make_unique<Frontier>(spec_.encoded_graph(), spec_.embed_model(model), options);
  else if (algorithm == "streaming")
    streaming_ = std::make_unique<Streaming>(spec_.encoded_graph(), spec_.embed_model(model), options);
  else throw std::invalid_argument("SettleExecutor algorithm must be frontier or streaming");
}

Result SettleExecutor::run(const Continuation& initial, const Tensor& values) {
  if (initial.cut < 0 || initial.cut % spec_.stride() || !initial.pending.empty())
    throw std::invalid_argument("SettleGraph run requires a complete position boundary");
  const auto start = initial.cut / spec_.stride();
  auto inputs = spec_.external(values, start);
  if (initial.batch_size != values.size(0)) throw std::invalid_argument("SettleGraph batch mismatch");
  const auto stop = (start + values.size(1)) * spec_.stride();
  return frontier_ ? frontier_->run(initial, inputs, stop, stop) : streaming_->run(initial, inputs, stop, stop);
}
}  // namespace tide
