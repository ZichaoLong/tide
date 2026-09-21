#include "lh_single_graph.h"
#include "tide/stream.h"
#include "tide/token_window.h"

namespace lh_iocortex {
namespace {
tide::Continuation initial(const tide::Graph& g) {
  tide::Continuation q; q.identity = g.identity; q.batch_size = Fixture::batch; return q;
}
std::vector<tide::External> inputs(const std::vector<tide::External>& xs, Index start, Index stop) {
  std::vector<tide::External> result;
  for (const auto& x : xs) if (start <= x.time && x.time < stop) result.push_back(x);
  return result;
}
void append(tide::Result& all, const tide::Result& part) {
  all.continuation = part.continuation;
  all.trace.insert(all.trace.end(), part.trace.begin(), part.trace.end());
  all.messages.insert(all.messages.end(), part.messages.begin(), part.messages.end());
  all.outputs.insert(all.outputs.end(), part.outputs.begin(), part.outputs.end());
}
}  // namespace
Index check_single_graph(const Fixture& f, int schedule, const std::vector<tide::External>& xs,
                         const ObservedSelector& is, const ObservedSelector& os,
                         const std::vector<at::Tensor>& logits) {
  SingleGraph single(f);
  tide::Options options; options.workers = schedule ? 3 : 1; options.packed = schedule == 2;
  tide::Streaming body(f.body, f.model, options), read(f.readout, f.read_model, options),
                  engine(single.graph, single.model, options);
  auto bq = initial(f.body), rq = initial(f.readout), q = initial(single.graph);
  std::vector<tide::External> global;
  for (auto x : xs) { x.time = single.body_clock.to_global(x.time); global.push_back(x); }
  const Index stop = logits.empty() ? single.body_clock.to_global(is.ticks.size()-1)+1
                                   : logits.size()*(single.layers+1);
  std::vector<tide::Output> pending; tide::Result all;
  for (Index cut = 1; cut <= stop; ++cut) {
    const auto start = bq.cut, body_cut = single.body_clock.cut(cut);
    auto br = body.run(bq, inputs(xs, bq.cut, body_cut), body_cut, body_cut); bq = br.continuation;
    pending.insert(pending.end(), br.outputs.begin(), br.outputs.end());
    auto r = engine.run(q, inputs(global, q.cut, cut), cut, cut); q = r.continuation; append(all, r);
    auto view = single.body_view(r, f.body);
    compare_result(view, br); compare_trace(f, view, is, os, start);
    tide::Result rr;
    const Index read_cut = single.read_clock.cut(cut);
    if (rq.cut < read_cut) {
      require(!pending.empty() || logits.empty(), "original Pronounce cannot certify an empty token window");
      auto external = pending.empty() ? std::vector<tide::External>{}
                                     : tide::token_inputs(pending, rq, single.layers, read_cut, bq.cut);
      rr = read.run(rq, external, read_cut, read_cut); rq = rr.continuation; pending.clear();
    } else rr.continuation = rq;
    // State/history quotient only: the external occurrence ledger is absent
    // from a single graph's internal readout wires and cannot be fabricated.
    auto expected = rq; expected.ledger.clear();
    compare_continuation(single.read_view(q, f.readout), expected);
    tide::Result out, target; out.outputs = r.outputs; target.outputs = rr.outputs;
    for (auto& o : out.outputs) {
      o.time = single.read_clock.to_local(o.time);
      if (!logits.empty()) {
        const auto& w = f.read_model.nodes[0];
        auto value = at::linear(o.value, w.extra.at("token_head"), w.extra.at("token_head_bias"));
        close(value, logits.at(o.time)[o.batch], "single graph original token logits mismatch");
        require(at::argmax(value).item<Index>() == at::argmax(logits.at(o.time)[o.batch]).item<Index>(),
                "single graph original greedy feedback mismatch");
      }
    }
    compare_result(out, target);
    out.outputs = single.buffer(q); target.outputs = pending; compare_result(out, target);
  }
  // A recorded greedy-input replay and every global cut must have the same
  // complete single-graph continuation, including unfinished token buffers.
  compare_result(all, engine.run(initial(single.graph), global, stop, stop));
  return stop;
}
}  // namespace lh_iocortex
