#include "lh_iocortex_fixture.h"
#include <algorithm>
#include <numeric>

namespace lh_iocortex {
namespace {
namespace AL = AccumulateLocal;
CSRMatrix<Index> adjacency(Index n, Index block) {
  std::vector<Index> ptr{0}, columns;
  for (Index v = 0; v < n; ++v) {
    std::vector<Index> row;
    if (v == 0) {
      for (Index target = 0; target < n; ++target) row.push_back(target);
      row.push_back(3);  // Parallel physical wires must remain distinct.
    } else if (v != n-1) row = {0, (v+block+1)%n};
    std::sort(row.begin(), row.end());
    columns.insert(columns.end(), row.begin(), row.end()); ptr.push_back(columns.size());
  }
  std::vector<Index> ids(columns.size());
  // Original physical IDs deliberately differ from CSR positions.
  for (size_t k = 0; k < ids.size(); ++k) ids[k] = ids.size()-1-k;
  return {n, n, ptr, columns, ids};
}
nlohmann::json al_config(const Policy& p) {
  const std::map<std::string, std::string> names{{"add", "add"}, {"sum", "add"}, {"mean", "average"},
    {"linear", "linear"}, {"active-softmax", "actsoftmax"}, {"all-softmax", "allsoftmax"}};
  return {{"chal", p.pool == "add" ? "add" : "attention"}, {"confluence", names.at(p.pool)},
    {"bias", p.bias()}, {"n_head", 2}, {"decay_rate", .03}, {"block_size", 2}};
}
tide::NodeWeights blank(at::TensorOptions opts) {
  const auto d = Fixture::width;
  return {at::zeros({d}, opts), at::zeros({d, d}, opts), at::zeros({d}, opts), at::zeros({d}, opts)};
}
void import_memory(tide::NodeWeights& w, const std::shared_ptr<torch::nn::Module>& module,
                   const Policy& p, Index id, Index slots) {
  const auto d = Fixture::width; auto opts = w.bias.options();
  if (p.pool == "add") { w.extra["add_retention"] = at::scalar_tensor(.97, opts); return; }
  auto a = std::dynamic_pointer_cast<AL::Attention>(module);
  require(bool(a), "IOCortex expected original Attention module");
  a->c_attn->weight.copy_(.04+at::sin(at::arange(3*d*d, opts).reshape({3*d, d})*.17+id*.11)*.02);
  a->c_proj->weight.copy_(.1+at::cos(at::arange(d*d, opts).reshape({d, d})*.21+id*.13)*.03);
  if (p.bias()) { a->c_attn->bias.fill_(.2+id*.003); a->c_proj->bias.fill_(.3+id*.007); }
  w.extra["fiber_qkv"] = a->c_attn->weight.t().clone();
  w.extra["fiber_out"] = a->c_proj->weight.t().clone();
  w.extra["fiber_qkv_bias"] = p.bias() ? a->c_attn->bias.clone() : at::zeros({3*d}, opts);
  w.extra["fiber_out_bias"] = p.bias() ? a->c_proj->bias.clone() : at::zeros({d}, opts);
  w.extra["fiber_decay"] = at::scalar_tensor(.03, opts);
  auto params = a->ptr2confluence->named_parameters();
  if (params.contains("weight")) {
    params["weight"].copy_(.2+at::arange(slots, opts)*.04);
    w.extra["fiber_pool"] = params["weight"].clone();
  }
}
tide::Node node(const Policy& p, Index region, const std::string& full) {
  tide::Node v{region}; v.memory = p.pool == "add" ? "lh-add-repeat-v1" : "lh-fiber-attention-"+p.pool+"-repeat-v1";
  v.query_heads = v.kv_heads = 2; v.full = full;
  return v;
}
}  // namespace

Fixture::Fixture(at::TensorOptions options, Policy p) : policy(std::move(p)), opts(options) {
  cfg = std::make_shared<GraphConfig>(nlohmann::json::parse(
      R"({"levelnum":2,"localnum":4,"base_num_or_hpnums":[0,1,2,8]})"));
  n = cfg->levelptr.back();
  const auto al = al_config(policy);
  const std::string act = policy.clear ? "relu" : "silu", norm = policy.lead ? "rms" : "identity";
  nlohmann::json cortex{{"emitD", width}, {"receiveD", width}, {"actfn", act}, {"norm", norm},
    {"bias", policy.bias()}, {"signalling", "linear"}, {"clear_after_activation", policy.clear}, {"chals", al}};
  nlohmann::json bridge{{"emitD", width}, {"receiveD", width}, {"bias", policy.bias()}, {"signalling", "linear"}};
  original = std::make_shared<IOCortexNet>(nlohmann::json{{"vocab_size", vocab}, {"n_layer", policy.layers()},
      {"input_", cortex}, {"output_", cortex}, {"iobridge_", bridge}, {"oibridge_", bridge}, {"pronounce", al}},
      cfg, adjacency(n, 0), adjacency(n, 1), adjacency(n, 2), adjacency(n, 3));
  original->to(opts.dtype().toScalarType());
  original->wte->weight.copy_(.5+at::sin(at::arange(vocab*width, opts).reshape({vocab, width})*.19)*.2);
  original->wte->weight[0].zero_();  // Present numerical-zero token, not absence.
  blocks = {original->inet.get(), original->onet.get(), original->iobridge.get(), original->oibridge.get()};
  const Index base = cfg->base_num, forced = policy.lead ? cfg->levelptr[cfg->levelptr.size()-3] : cfg->levelptr[cfg->levelptr.size()-2];
  const Index points = cfg->levelptr[cfg->levelptr.size()-2];
  for (Index c = 0; c < 2; ++c) {
    for (Index r = 0; r < base; ++r) body.regions.push_back({2, true, true, "proposal", "lh-count-affect-v1"});
    body.regions.push_back({forced, true, true, "proposal", "count-v1"});
    auto& original_cortex = c ? *original->onet : *original->inet;
    for (Index v = 0; v < n; ++v) {
      Index r = v < forced ? base : v < points ? v-forced : (v-points)/cfg->localnum;
      auto nv = node(policy, c*(base+1)+r, "lh-"+act+"-"+norm+"-v1");
      nv.clear = policy.clear; nv.emission = "slot_affine"; nv.readout = "norm-fp64-v1";
      body.nodes.push_back(nv); auto w = blank(opts);
      import_memory(w, original_cortex.chal.anymodules[v].ptr(), policy, c*n+v, original_cortex.chalns[v]);
      if (norm != "identity") {
        auto params = original_cortex.norm.anymodules[v].ptr()->named_parameters();
        params["weight"].copy_(.8+at::arange(width, opts)*.05+v*.01);
        w.extra["lh_norm_weight"] = params["weight"].clone();
      }
      model.nodes.push_back(w);
    }
  }
  body.inputs = {0}; body.outputs = {n}; body.layout = tide::PortLayout{};
  auto& layout = *body.layout;
  std::vector<Index> in(2*n, 0), out(2*n, 0);
  const std::array<Index, 4> src{0, n, 0, n}, dst{0, n, n, 0};
  for (Index block = 0; block < 4; ++block) {
    auto& b = *blocks[block]; const Index offset = body.edges.size(); edge_offsets[block] = offset;
    body.edges.resize(offset+b.A.nnz); layout.edge_source.resize(body.edges.size()); layout.edge_target.resize(body.edges.size());
    for (Index v = 0; v < n; ++v) {
      auto params = b.nets.anymodules[v].ptr()->named_parameters();
      if (b.targetns(v)) {
        auto shape = params["weight"].sizes();
        params["weight"].copy_(.02+at::sin(at::arange(params["weight"].numel(), opts).reshape(shape)*.17+v*.41+block)*.005);
        if (policy.bias()) params["bias"].copy_(.05+at::arange(params["bias"].numel(), opts)*.003+block*.01+v*.002);
      }
      for (Index k = b.A.indptr[v]; k < b.A.indptr[v+1]; ++k) {
        const auto edge = offset+b.A.data[k], slot = out[src[block]+v]++, local = k-b.A.indptr[v];
        body.edges[edge] = {src[block]+v, dst[block]+b.A.indices[k], 1}; layout.edge_source[edge] = slot;
        auto& w = model.nodes[src[block]+v];
        w.extra["emit_w_"+std::to_string(slot)] = params["weight"].slice(0, local*width, (local+1)*width).t().clone();
        w.extra["emit_b_"+std::to_string(slot)] = policy.bias() ? params["bias"].slice(0, local*width, (local+1)*width).clone() : at::zeros({width}, opts);
      }
      for (Index k = b.AT.indptr[v]; k < b.AT.indptr[v+1]; ++k)
        layout.edge_target[offset+b.AT.data[k]] = in[dst[block]+v]++;
    }
  }
  layout.input = {in[0]++}; layout.output = {out[n]++};
  auto output_slot = std::to_string(layout.output[0]);
  model.nodes[n].extra["emit_w_"+output_slot] = at::eye(width, opts);
  model.nodes[n].extra["emit_b_"+output_slot] = at::zeros({width}, opts);
  body.compile();
  model.input_scale = model.output_scale = {at::ones({}, opts)};
  model.agg_scale.assign(body.edges.size(), at::ones({}, opts)); model.edge_scale = model.agg_scale;
  readout.nodes = {node(policy, 0, "lh-identity-identity-v1")}; readout.regions = {{1}};
  readout.inputs.assign(policy.layers(), 0); readout.outputs = {0}; readout.compile();
  auto rw = blank(opts); import_memory(rw, original->pronounce->chal.ptr(), policy, 2*n, policy.layers());
  original->pronounce->lm_head->weight.copy_(at::cos(at::arange(vocab*width, opts).reshape({vocab, width})*.23)*.3);
  if (policy.bias()) original->pronounce->lm_head->bias.copy_(at::arange(vocab, opts)*.02);
  rw.extra["token_head"] = original->pronounce->lm_head->weight.clone();
  rw.extra["token_head_bias"] = policy.bias() ? original->pronounce->lm_head->bias.clone() : at::zeros({vocab}, opts);
  read_model.nodes = {rw}; read_model.input_scale.assign(policy.layers(), at::ones({}, opts));
  read_model.output_scale = {at::ones({}, opts)};
}
}  // namespace lh_iocortex
