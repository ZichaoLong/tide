"""Active-scale packets must certify reachability independently of model size."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('flow_topology', Path(__file__).parents[1]/'scripts/flow_topology.py')
flow=importlib.util.module_from_spec(spec); spec.loader.exec_module(flow)


@pytest.mark.parametrize('layers,width', [(4,4),(15,32),(128,32)])
def test_all_nodes_reachable_and_useful(layers,width):
    graph=flow.ranked_graph(layers=layers,region_width=width,fanout=2,local_span=width,skip=1)
    n=layers*width
    assert graph['counts']['reachable_nodes']==graph['counts']['output_reachable_nodes']==n
    # Every column has an independently identifiable adjacent path throughout.
    assert all([r*width+c,(r+1)*width+c,1] in graph['edges'] for r in range(layers-1) for c in range(width))


def test_packet_roundtrip_and_tampering():
    packet=flow.make_packet(graph=flow.ranked_graph(layers=4,region_width=4,fanout=2,local_span=2))
    assert flow.validate_packet(packet) is packet
    assert packet['families']==['settle','timed-dag','pdg']
    assert flow.native_text(packet).startswith('TIDE_COMPLETE_FLOW_1\n16 4 ')
    altered=copy.deepcopy(packet); altered['workload']['tokens']+=1
    with pytest.raises(ValueError,match='hash'): flow.validate_packet(altered)


def test_dormant_region_is_rejected():
    graph=flow.ranked_graph(layers=4,region_width=4,fanout=2,local_span=2)
    graph['nodes']+=1
    with pytest.raises(ValueError,match='every body node'): flow.graph_counts(graph)


def test_delay_stress_is_not_mislabeled_settle():
    graph=flow.ranked_graph(layers=4,region_width=4,fanout=2,local_span=2,delayed=True)
    packet=flow.make_packet(graph=graph)
    assert packet['families']==['timed-dag','pdg']
    assert packet['workload']['stride']>graph['counts']['longest_body_delay']


@pytest.mark.parametrize('value',[True,2.0,-1,2**63])
def test_exact_integer_contract(value):
    with pytest.raises(ValueError):flow.ranked_graph(layers=value)


def test_scale_counts_grow_with_actual_wires():
    small=flow.ranked_graph(layers=4)
    large=flow.ranked_graph(layers=15)
    assert len(large['edges'])>4*len(small['edges'])
    add=flow.make_packet(graph=large,width=2048,batch=512,tokens=12,vocab=50304)
    attention=flow.make_packet(graph=large,memory='attention',width=2048,batch=512,tokens=12,vocab=50304)
    assert add['counts']['reachable_nodes']==480
    assert attention['counts']['parameters']-add['counts']['parameters']==4*480*2048**2
