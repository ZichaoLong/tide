"""Execution clients for frozen workloads; position cuts are explicit."""
import torch
import _tide_native as core
from tidegraph import Continuation
from tidegraph.native import Native
from tidegraph.native_records import from_continuation, to_continuation, window_records
from tidegraph.records import Result
from tidegraph.reference import run as streaming
from tidegraph.settle import run as settle
from tidegraph.specialized import settle_layered
from foundation_workloads import initialize, inputs_for


class LegacyExecution:
    def __init__(self,config,variant,workers=4,trace=False):
        self.config,self.variant=config,variant
        self.graph,self.spec,self.model=initialize(config,'linear' if variant.endswith('efficient') else 'input')
        self.values,self.external,self.stride,self.lengths=inputs_for(config,self.graph)
        self.engine=None; self.compiled=None; self.trace=trace
        self.options=dict(packed=not variant.endswith('scalar'),trace=trace,
            workers=workers if any(word in variant for word in ('workers','transport','frontier','settle','diamond')) else 1,
            mode='hst' if config.get('training_window') else 'hard', prefill=not variant.endswith('step'))
        if variant.endswith('single'): self.options['attention_packing']='single'
        if variant.endswith('transport') or variant.endswith('efficient'):
            self.options.update(packed_sources=True,batch_next=True,parallel_regions=True,compact_events=True,defer_state_release=True)
        if variant.endswith('efficient'):
            self.options.update(attention_packing='single',fiber_pooling='csr',fiber_cache='owned',attention_layout='head')
        if variant.startswith('native-settle'):
            if self.spec is None: raise ValueError('native Settle requires SettleGraph')
            body=Native(self.graph,self.model,**self.options)
            self.compiled=core.SettleGraph(body.compiled,self.spec.ranks)
            options=core.Options()
            for key,value in self.options.items(): setattr(options,key,value)
            self.engine=core.SettleExecutor(self.compiled,body.weights,options)
        elif variant=='encoded-frontier':
            if self.spec is None: raise ValueError('encoded frontier requires SettleGraph')
            self.eg,self.em=self.spec.embed(self.model)
            self.engine=Native(self.eg,self.em,algorithm='frontier',**self.options)
        elif variant.startswith('native'):
            if self.spec: raise ValueError('use a native Settle or encoded variant for SettleGraph')
            algorithm='diamond' if variant=='native-diamond' else 'frontier' if 'frontier' in variant else 'streaming'
            self.engine=Native(self.graph,self.model,algorithm=algorithm,**self.options)
        elif variant not in {'python-stream','python-settle','python-layered'}:
            raise ValueError('unknown benchmark variant')
        self.reset()

    def reset(self):
        self.q=Continuation(self.graph.identity,self.config['batch']);self.position=0
        if self.compiled:
            self.eq=core.Continuation();self.eq.identity=self.compiled.encoded_graph.identity;self.eq.batch_size=self.q.batch_size
        elif self.variant=='encoded-frontier': self.eq=self.spec.embed_initial(self.q,self.eg)

    def advance(self,stop):
        if not self.position <= stop <= self.config['sequence']: raise ValueError('invalid explicit position cut')
        if self.spec:
            x=self.values[:,self.position:stop]
            if self.compiled:
                encoded=self.engine.run(self.eq,x);self.eq=encoded.continuation
                body=self.compiled.project(encoded)
                result=Result(from_continuation(self.graph,body.continuation),*window_records(body))
                # Keep encoded counters: body projection is a view, boundary work is real.
                result.stats=dict(encoded.stats)
            elif self.variant=='encoded-frontier':
                xs=self.spec.external(x,self.position,encoded=True);end=stop*self.spec.stride
                encoded=self.engine.run(self.eq,xs,end,sealed_until=end);self.eq=encoded.continuation
                result=self.spec.project(encoded)
            else:
                fn=settle_layered if self.variant=='python-layered' else settle
                result=fn(self.spec,self.model,self.q,x,mode=self.options['mode'])
        else:
            end=self.stride*stop
            xs=[x for x in self.external if self.q.cut<=x.time<end]
            result=self.engine.run(self.q,xs,end,sealed_until=end) if self.engine else streaming(
                self.graph,self.model,self.q,xs,end,sealed_until=end,mode=self.options['mode'])
        self.position=stop;self.q=result.continuation
        return result

    def detach(self):
        self.q=self.q.detach()
        if self.compiled:
            # Identity-only adapter for tensor-preserving continuation conversion.
            from types import SimpleNamespace
            graph=SimpleNamespace(identity=self.compiled.encoded_graph.identity)
            q=from_continuation(graph,self.eq).detach()
            self.eq=to_continuation(core,graph,self.compiled.encoded_graph,q)
        elif self.variant=='encoded-frontier': self.eq=self.eq.detach()


def Execution(config, variant, workers=4, trace=False):
    if config.get('execution_schema') == 'v2':
        from foundation_execution_v2 import Execution as V2
        return V2(config, variant, workers, trace)
    return LegacyExecution(config, variant, workers, trace)


def loss(result):
    # Fixed mean roots keep training scale comparable across window lengths.
    values=[v.square().mean() for _,_,_,v in result.outputs]
    values += [s.value.square().mean()*.1 for s in result.continuation.states.values()]
    values += [v.square().mean()*.01 for s in result.continuation.states.values() for v in s.slots.values() if v.numel()]
    if not values: raise ValueError('training window has no differentiable root')
    return sum(values)/len(values)


def observations(execution,result):
    q=result.continuation; stats=dict(result.stats)
    if result.trace:
        stats.update(candidate_events=len(result.trace),selected_events=sum(e['active'] for e in result.trace),
            source_rows=sum(len(e['fiber']) for e in result.trace),visited_edges=len(result.messages),
            body_candidate_events=len(result.trace),body_selected_events=sum(e['active'] for e in result.trace))
    candidates=stats.get('candidate_events',0);selected=stats.get('selected_events',0)
    values=[s.value for s in q.states.values()]+[v for _,_,_,v in result.outputs]
    slots=[v for s in q.states.values() for v in s.slots.values()]
    if any(not torch.isfinite(v).all() for v in values+slots+[m.value for m in q.pending]):
        raise RuntimeError('nonfinite workload output/state/slots/pending')
    return {'stats':stats,'logical':{'Aggregate':candidates,'Upd':stats.get('body_candidate_events',candidates),
                'Next':candidates,'Read':candidates,'Full':selected},
            'touched_state_owners':len(q.states),'touched_body_nodes':len({n for _,n in q.states}),
            'edge_visits':stats.get('visited_edges',len(result.messages)),
            'pending_messages':len(q.pending),'outputs':len(result.outputs),'input_ledger_entries':len(q.ledger),
            'cache_tensor_elements':sum(v.numel() for v in slots),
            'max_cache_rows':max([len(s.slots['key']) for s in q.states.values() if 'key' in s.slots]+[0]),
            'output_checksum':sum(float(v.detach().double().sum()) for _,_,_,v in result.outputs),
            'state_checksum':sum(float(s.value.detach().double().sum()) for s in q.states.values()),
            'effective_input_positions':sum(execution.lengths)}
