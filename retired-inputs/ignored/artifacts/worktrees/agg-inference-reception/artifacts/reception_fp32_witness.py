"""Bounded development witness; the scalar replay is independent Decimal math."""
from dataclasses import replace
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tests"))
from tide import TideModel
from benchmarks.continuous import nested_comparison
from scripts.job import atomic
from scripts.qualify_precision import fingerprint
from test_reception_execution import config
from test_training_optimizations import update


def scalar_replay(before, after, index):
    def exact(value):
        return Decimal.from_float(float(value))
    with localcontext() as context:
        context.prec = 80
        lr, decay, b1, b2, eps = map(Decimal, ("0.0001", "0.01", "0.9", "0.999", "0.00000001"))
        g = exact(after["gradient"][index])
        m = b1*exact(before["exp_avg"][index]) + (1-b1)*g
        v = b2*exact(before["exp_avg_sq"][index]) + (1-b2)*g*g
        step = int(after["step"])
        def value(moment, square):
            normalized = (moment/(1-b1**step)) / ((square/(1-b2**step)).sqrt()+eps)
            return exact(before["parameter"][index])*(1-lr*decay) - lr*normalized
        predicted = value(m,v)
        recorded_moments = value(exact(after["exp_avg"][index]), exact(after["exp_avg_sq"][index]))
        actual = exact(after["parameter"][index])
        return dict(from_prior_slots_and_gradient=float(predicted), from_recorded_new_slots=float(recorded_moments),
                    actual=float(actual), absolute_error=float(abs(actual-predicted)),
                    recorded_slots_absolute_error=float(abs(actual-recorded_moments)))


torch.set_num_threads(1)
torch.manual_seed(519)
cfg = config("incoming_light_64", "settle")
models = [TideModel(cfg), TideModel(replace(cfg,reception_qkv_backend="fused",
                  reception_mask_padding=True, checkpoint_reception=True))]
models[1].load_state_dict(models[0].state_dict())
optimizers = [torch.optim.AdamW(model.parameters(), lr=1e-4) for model in models]
states = [None, None]
name = "regional.4.second.bias"
record = dict(source=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              production_tree=subprocess.check_output(["git","rev-parse","HEAD:src/tide"],text=True).strip(),
              harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seed=519,
              initial_parameters_sha256=fingerprint(models[0].state_dict()),
              configs=[model.config.to_dict() for model in models], target_parameter=name,
              threshold=dict(atol=1e-6,rtol=2e-3), updates=[],
              scope="CPU FP32 bounded three-step synthetic fixture; scalar Decimal update replay explains selected target coordinates only")
path = Path(sys.argv[1])
if path.exists():
    raise FileExistsError(path)
for step in range(3):
    inputs = torch.randn(1,5,64)
    observations, before, after = [], [], []
    for i,(model,optimizer) in enumerate(zip(models,optimizers)):
        parameter = dict(model.named_parameters())[name]
        slot = optimizer.state.get(parameter, {})
        before.append(dict(parameter=parameter.detach().tolist(),
                           exp_avg=slot.get("exp_avg", torch.zeros_like(parameter)).tolist(),
                           exp_avg_sq=slot.get("exp_avg_sq", torch.zeros_like(parameter)).tolist()))
        states[i], gradients, results = update(model,optimizer,states[i],inputs)
        slot = optimizer.state[parameter]
        after.append(dict(parameter=parameter.detach().tolist(), gradient=parameter.grad.tolist(),
                          exp_avg=slot["exp_avg"].tolist(), exp_avg_sq=slot["exp_avg_sq"].tolist(),
                          step=int(slot["step"])))
        observations.append(dict(state=states[i], gradients=gradients, results=results,
                                 parameters=model.state_dict(), optimizer=optimizer.state_dict()))
    report = nested_comparison(*observations)
    index=max(range(len(after[0]["parameter"])),key=lambda j: abs(after[0]["parameter"][j]-after[1]["parameter"][j]))
    replays=[scalar_replay(a,b,index) for a,b in zip(before,after)]
    predicted_delta=replays[1]["from_prior_slots_and_gradient"]-replays[0]["from_prior_slots_and_gradient"]
    actual_delta=after[1]["parameter"][index]-after[0]["parameter"][index]
    row=dict(step=step+1,input_sha256=fingerprint(inputs),comparison=report,before=before,after=after,
             coordinate=index,replays=replays,actual_delta=actual_delta,decimal_predicted_delta=predicted_delta,
             delta_absolute_error=abs(actual_delta-predicted_delta))
    record["updates"].append(row)
    atomic(path,record)
    print(json.dumps(dict(step=step+1,comparison=report,coordinate=index,
          gradient=[item["gradient"][index] for item in after],actual_delta=actual_delta,
          decimal_predicted_delta=predicted_delta,delta_absolute_error=abs(actual_delta-predicted_delta))),flush=True)
