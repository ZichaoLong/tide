"""Optional exact common-function evaluation inside one certified region block.

A program must explicitly supply a pure-function sharing key. Keys are scoped
to this invocation; no value comparison, trajectory replay or persistent cache
is used. Each event still owns its complete physical sources and Content view.
"""
from .aggregate import request, validate, AggregateResult, evaluate
from .content import Content


def evaluate_shared(graph, model, by_node, stats, *, packed, aggregate_autograd):
    cache, representatives, copies = {}, {}, []
    for node, events in by_node.items():
        program = model.nodes[node].aggregate_program
        key_for = getattr(program, "sharing_key", None)
        if key_for is None:
            representatives[node] = events
            continue
        selected = []
        for event in events:
            req = request(graph, model, event["fiber"])
            key = None if key_for is None else key_for(req)
            key = None if key is None else (type(program), key)
            if key is not None and key in cache:
                copies.append((event, req, cache[key]))
            else:
                selected.append(event)
                if key is not None:
                    cache[key] = event
        if selected:
            representatives[node] = selected
    for node, events in representatives.items():
        evaluate(graph, model, events, packed=packed, aggregate_autograd=aggregate_autograd)
        stats["aggregate_calls"] = stats.get("aggregate_calls", 0)+(1 if packed else len(events))
    for event, req, source in copies:
        result = AggregateResult(source["content"], source["contributions"])
        validate(result, req)
        event.update(content=result.value, contributions=result.contributions)
        event["_content"] = Content(result.value, req.sources, result.contributions)
    stats["aggregate_computed_events"] = sum(map(len, representatives.values()))
    stats["aggregate_reused_events"] = len(copies)
