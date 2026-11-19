"""Small current-instance integration, parameterization and continuation checks."""
import json
import torch
import pytest

from tide import ModelConfig, TopologyConfig, TideModel, detach_state, save_checkpoint, load_checkpoint


@pytest.mark.parametrize("aggregation,sharing,allocation",[
    ("source_residual","independent","quad"),
    ("source_residual","region_shared","shannon"),
    ("source_residual","layer_shared","pool"),
    ("incoming_light_64","independent","pool"),
    ("incoming_light_128","region_shared","quad"),
    ("incoming_wide","layer_shared","shannon"),
])
def test_fc_profiles_forward_backward_and_continue(aggregation,sharing,allocation):
    torch.manual_seed(72)
    cfg=ModelConfig(topology=TopologyConfig(n=2,layers=2),d=64,
                    aggregation=aggregation,sharing=sharing,allocation=allocation)
    model=TideModel(cfg).double()
    x=torch.randn(2,5,64,dtype=torch.float64,requires_grad=True)
    whole=model(x)
    a=model(x[:,:2]); b=model(x[:,2:],a.state)
    torch.testing.assert_close(torch.cat((a.readout(),b.readout()),1),whole.readout(),rtol=2e-9,atol=2e-9)
    whole.readout().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for memory in whole.state.memories:
        assert (memory["K"].weight==1).all()  # All FC candidates write, not only Full.


@pytest.mark.parametrize("topology",[
    TopologyConfig(kind="lattice",n=2,depth=1),
    TopologyConfig(kind="exchange",n=2,depth=1,rounds=2),
    TopologyConfig(kind="exchange",n=2,depth=1,rounds=1,hubs=1),
    TopologyConfig(kind="multiscale",n=2,profile="compact-5-dd4",hubs=1),
    TopologyConfig(kind="multiscale",n=2,profile="compact-5-dd4",hubs=1,partition="singleton"),
    TopologyConfig(kind="multiscale",n=2,profile="compact-5-dd4",hubs=1,partition="mixed"),
    TopologyConfig(kind="multiscale",n=2,profile="compact-5-dd4",hubs=1,timing="timed"),
])
def test_hb_window_continuation(topology):
    torch.manual_seed(12)
    m=TideModel(ModelConfig(topology=topology,d=64,sharing="layer_shared")).double()
    x=torch.randn(1,4,64,dtype=torch.float64)
    with torch.no_grad():
        y=m(x); first=m(x[:,:2]); second=m(x[:,2:],first.state)
    torch.testing.assert_close(torch.cat((first.readout(),second.readout()),1),y.readout(),rtol=1e-8,atol=1e-8)
    torch.testing.assert_close(torch.cat((first.strengths,second.strengths),2),y.strengths,rtol=1e-8,atol=1e-8)
    if topology.timing=="timed":
        assert first.state.pending
        assert any((layer.strengths>0).any() for layer in first.state.pending)


def test_parameter_count_and_fc_shared_identity():
    d,n,layers=64,2,2
    m=TideModel(ModelConfig(topology=TopologyConfig(n=n,layers=layers),d=d))
    expected=layers*n*(17.5*d*d+2.5*d+1+d/8)+layers*(2*d*d+2*d)+d*d
    assert sum(p.numel() for p in m.parameters())==expected
    torch.manual_seed(3)
    a=TideModel(ModelConfig(topology=TopologyConfig(n=n,layers=layers),d=d,sharing="region_shared"))
    torch.manual_seed(3)
    b=TideModel(ModelConfig(topology=TopologyConfig(n=n,layers=layers),d=d,sharing="layer_shared"))
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa,pb,rtol=0,atol=0)


def test_adam_checkpoint_and_next_window(tmp_path):
    torch.manual_seed(84)
    model=TideModel(ModelConfig(topology=TopologyConfig(n=2,layers=2),d=64)).double()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    state=None
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        result=model(torch.randn(2,2,64,dtype=torch.float64),state)
        result.readout().square().mean().backward(); optimizer.step()
        state=detach_state(result.state)
    path=tmp_path/"model.pt"
    save_checkpoint(path,model,state=state,optimizer=optimizer)
    restored,rs,os=load_checkpoint(path)
    opt=torch.optim.AdamW(restored.parameters(),lr=1e-4); opt.load_state_dict(os)
    x=torch.randn(2,2,64,dtype=torch.float64)
    for m,o,s in ((model,optimizer,state),(restored,opt,rs)):
        o.zero_grad(set_to_none=True)
        m(x,s).readout().square().mean().backward(); o.step()
    for a,b in zip(model.parameters(),restored.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_configuration_json():
    cfg=ModelConfig(topology=TopologyConfig(kind="multiscale",profile="compact-9",hubs=4))
    assert ModelConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))==cfg


def test_custom_configuration_json_and_budget_validation():
    cfg=ModelConfig(topology=TopologyConfig(kind="multiscale",profile="compact-5-dd4",n=8,hubs=4,
                    hub_mask=((True,),(False,False),(True,False),(True,False),(False,)),probes=(1,1,1)))
    assert ModelConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))==cfg
    # Configurations may use tiny formula-witness graphs, but the public
    # ordinary budget keeps the registered menu, including its 64 upper end.
    for budget in (3,128,True,1.5):
        with pytest.raises(ValueError):
            ModelConfig(topology=TopologyConfig(n=128),budget=budget)


@pytest.mark.parametrize("memory,topology,profile,sharing",[
    ("S",TopologyConfig(n=2,layers=2),"source_residual","independent"),
    ("S",TopologyConfig(n=2,layers=2),"incoming_wide","region_shared"),
    ("X",TopologyConfig(kind="exchange",n=2,depth=1,hubs=1),"incoming_light_64","layer_shared"),
    ("X",TopologyConfig(kind="multiscale",n=2,profile="compact-5-dd4",hubs=1,timing="timed"),"source_residual","region_shared"),
    ("S",TopologyConfig(kind="multiscale",n=2,profile="compact-5-dd4",hubs=1,timing="timed",partition="mixed"),"source_residual","independent"),
])
def test_recurrent_and_mixed_complete_model(memory,topology,profile,sharing):
    torch.manual_seed(453)
    m=TideModel(ModelConfig(topology=topology,d=64,memory=memory,aggregation=profile,sharing=sharing)).double()
    x=torch.randn(1,4,64,dtype=torch.float64,requires_grad=True)
    whole=m(x); a=m(x[:,:2]); b=m(x[:,2:],a.state)
    torch.testing.assert_close(whole.readout(),torch.cat((a.readout(),b.readout()),1),rtol=2e-7,atol=2e-7)
    whole.readout().square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
