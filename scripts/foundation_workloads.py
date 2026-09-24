"""Frozen topology/module definitions shared by smoke, timing and scale preflight."""
import math
import torch
from tidegraph import Continuation, Edge, External, Graph, Node, Region
from tidegraph.fiber_attention import PROFILE
from tidegraph.ops import Model
from tidegraph.settle import SettleGraph


VARIANTS = {
    'P01':['python-stream','native-stream-scalar','native-stream-packed'],
    'P02':['native-stream-packed'],
    'P03':['native-stream-packed','native-stream-workers','native-stream-transport'],
    'T01':['native-stream-packed','native-frontier','native-diamond'],
    'T02':['native-frontier','native-frontier-step'],
    'S01':['python-settle','python-layered','native-settle','encoded-frontier'],
    'S02':['native-settle','native-settle-step'],
    'A01':['native-stream-packed','native-frontier'],
    'A02':['native-frontier','native-frontier-single','native-frontier-efficient'],
    'M01':['native-stream-packed','native-frontier','native-frontier-step'],
    'M02':['native-stream-packed','native-frontier','native-frontier-step'],
    'TR01':['native-stream-scalar','native-stream-packed','native-frontier'],
}


def graph_for(config):
    n = config['body_nodes']; topology = config['topology']; family = config['graph']
    module = config['module']; full = 'swiglu' if 'swiglu' in module else 'tanh'
    memory = {'gated-delta':'delta','fiber-attention':PROFILE,'fiber-attention-swiglu':PROFILE,'ssm-swiglu':'ssm',
              'add-swiglu':'lh-add-repeat-v1',
              'attention-gqa':'attention','attention-gqa-window128':'attention'}.get(module,module)
    heads, kv = (4,1) if 'gqa' in module else (4,4)
    window = 128 if 'window128' in module else 0
    if topology.startswith('sparse-ring'):
        owners = list(range(n)); edges = [Edge(v,(v+1)%4,1) for v in range(4)]
        inputs, outputs, budgets, ranks = (0,), (3,), [1]*n, None
    elif topology == 'source-regions':
        owners = [v//2 for v in range(n)]; edges = [Edge(v,v,1) for v in range(8)]
        inputs = tuple(v for v in range(8) for _ in range(2)); outputs = tuple(range(8))
        budgets, ranks = [1]*(n//2), None
    elif topology == 'diamond':
        if n != 4: raise ValueError('diamond requires four body nodes')
        owners, edges = [0,1,1,2], [Edge(a,b,1) for a,b in ((0,1),(0,2),(1,3),(2,3))]
        inputs, outputs, budgets, ranks = (0,), (3,), [1,2,1], (1,2,3)
    elif topology in {'four-regions','four-layers','large-layered'}:
        if topology == 'large-layered':
            width = config['nominal_selection_denominator']; layers = n//width
            if n % width or layers < 4: raise ValueError('large nodes must form at least four complete regions')
            active_layers = 4
        else: layers = 4; width = n//4; active_layers = 4
        owners = [v//width for v in range(n)]
        edges = [Edge(a,b,1) for r in range(active_layers-1) for a in range(r*width,(r+1)*width)
                 for b in range((r+1)*width,(r+2)*width)]
        inputs = tuple(range(width)); outputs = tuple(range((active_layers-1)*width,active_layers*width))
        budgets = [1 if topology == 'large-layered' else max(1,width//2) if topology == 'four-regions' else width]*layers
        ranks = tuple(range(1,layers+1))
    elif topology.startswith('chain'):
        owners = list(range(n)); edges = [Edge(v,v+1,1) for v in range(n-1)]
        inputs, outputs, budgets, ranks = (0,), (n-1,), [1]*n, tuple(range(1,n+1))
    else: raise ValueError('unknown frozen topology')
    nodes = tuple(Node(r,memory=memory,full=full,query_heads=heads,kv_heads=kv,window=window,
                       clear=topology == 'four-regions' and r == 3) for r in owners)
    graph = Graph(nodes,tuple(edges),tuple(Region(b) for b in budgets),inputs,outputs)
    spec = SettleGraph(graph,ranks) if family == 'settle' else None
    if family != 'pdg': graph.topological_order()
    return graph,spec


def initialize(config, projection_layout='input'):
    graph,spec = graph_for(config); d=config['width']
    model=Model(graph,width=d,dtype=getattr(torch,config['dtype']),projection_layout=projection_layout)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 2: p.mul_(1/(.15*math.sqrt(d)))
        for group in (model.input_scale,model.output_scale,model.agg_scale,model.edge_scale):
            for p in group: p.fill_(.8)
    return graph,spec,model


def inputs_for(config, graph):
    b,t,d = config['batch'],config['sequence'],config['width']
    values = (torch.sin(torch.arange(b*t*d,dtype=getattr(torch,config['dtype']))*.019).reshape(b,t,d)*.02)
    lengths = [t-(i%4)*(t//8) if 'ragged' in config['topology'] else t for i in range(b)]
    # An explicit sealed application clock, not an inferred occurrence ledger.
    stride = 1 if config['graph']=='pdg' and config['topology']!='diamond' else 6 if config['topology']=='large-layered' else len(graph.regions)+2
    external = [External(sample,port,pos,stride*pos,values[sample,pos])
                for sample in range(b) for port in range(len(graph.inputs)) for pos in range(lengths[sample])]
    return values,external,stride,lengths


def count(config):
    """Exact owners for these unshared profiles without allocating the large model."""
    graph,spec = graph_for(config); d=config['width']; node=graph.nodes[0]
    # Probe only one node's actual initializer; count graph-owned scalar scales separately.
    one=Graph((Node(0,memory=node.memory,full=node.full,query_heads=node.query_heads,
                    kv_heads=node.kv_heads,window=node.window),),(),(Region(1),),(0,),(0,))
    with torch.device('meta'):
        per_node=sum(p.numel() for p in Model(one,width=d,dtype=torch.float32).nodes[0].parameters())
    parameters=len(graph.nodes)*per_node+len(graph.inputs)+len(graph.outputs)+2*len(graph.edges)
    return {'body_nodes':len(graph.nodes),'regions':len(graph.regions),'edges':len(graph.edges),
            'input_ports':len(graph.inputs),'output_ports':len(graph.outputs),'parameters':parameters,
            'parameters_per_node':per_node,'parameter_sharing':'none; each body node owns its complete module',
            'encoded_nodes':len(graph.nodes)+2 if spec else len(graph.nodes)}


def estimate(config):
    counts=count(config); d,b,t=config['width'],config['batch'],config['sequence']; item=8 if config['dtype']=='float64' else 4
    touched = config.get('nominal_touched_nodes',min(counts['body_nodes'],4*config.get('nominal_selection_denominator',1)))
    memory=config['module']; heads=4
    # Conservative peak: values + constructor transients, KV and score matrices,
    # three simultaneous event snapshots, metadata, gradients/AdamW when used.
    parameters=counts['parameters']*item; train=config.get('training_window') is not None
    kv = b*t*touched*d*2*item*(3 if 'fiber-attention' in memory else 1) if 'attention' in memory else 0
    scores = b*heads*t*t*item*3 if 'attention' in memory else 0
    matrix = b*t*touched*d*d*item*3 if memory in {'linear','gated-delta'} else 0
    events=b*t*touched*(d*item*12+4096)
    return {**counts,'parameter_bytes':parameters,'gradient_bytes':parameters if train else 0,
            'optimizer_bytes':2*parameters if train else 0,'kv_bytes_upper':kv,'score_bytes_upper':scores,
            'matrix_state_bytes_upper':matrix,'event_bytes_upper':events,
            'estimated_peak_bytes':parameters*(6 if train else 2)+kv+scores+matrix+events+2**30}


def large_config(preset, family, nodes=None):
    d,denom = preset['width'],preset['nominal_selection_denominator']
    config=dict(id=preset['id'],graph=family,topology='large-layered',width=d,batch=preset['batch'],
                sequence=6,body_nodes=4*denom,module=preset.get('module','fiber-attention-swiglu'),dtype='float32',training_window=None,
                nominal_selection_denominator=denom,nominal_touched_nodes=4*denom,modes=['nograd-forward'])
    per=count(config)['parameters_per_node']; target=preset.get('reference_parameters',8500000000)
    config['body_nodes']=nodes or max(4*denom,round(target/per/denom)*denom)
    return config
