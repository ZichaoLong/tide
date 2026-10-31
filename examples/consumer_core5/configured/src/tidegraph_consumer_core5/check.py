"""Independent tiny task formula/RNG checks, separate from graph qualification."""
import argparse
import io
import json
from types import SimpleNamespace
import torch
from tidegraph import External
from tidegraph.runtime import resolve_device
from .task import create_task


def check(device="cpu"):
    actual_device, _ = resolve_device(device)
    task_config = dict(seed=7, optimizer=dict(kind="sgd", lr=.0001, momentum=.3, weight_decay=.01))
    plan = dict(width=4, span=8)
    make = lambda: create_task(task=task_config, plan=plan, options={})
    task = make()
    task.module.to(actual_device)
    root = torch.ones(5)
    templates = [External(0, 0, 0, 0, root), External(0, 1, 0, 1, root)]
    inputs = task.inputs(templates, 0, training=True)
    if inputs[0].value is not inputs[1].value:
        raise AssertionError("physical ports lost their common input root")
    payload = torch.tensor([.5, 1., 2., 3., 4.], device=actual_device, requires_grad=True)
    result = SimpleNamespace(outputs=[(0, 0, 0, payload)])
    loss = task.loss(result, training=True)
    weight, bias = task.module.weight, task.module.bias
    # Scalar target is zero at this coordinate. This anchor uses no task loss.
    prediction = ((payload[0]*payload[1:])*weight[0]).sum()+bias[0]
    expected = prediction.square()*.001
    torch.testing.assert_close(loss, expected)
    owners = (payload, weight, bias)
    actual_grad = torch.autograd.grad(loss, owners, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, owners)
    for actual, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, want)
    buffer = io.BytesIO()
    torch.save(task.state_dict(), buffer)
    buffer.seek(0)
    restored = make()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    uninterrupted, resumed = task.inputs(templates, 1), restored.inputs(templates, 1)
    for left, right in zip(uninterrupted, resumed):
        if (left.batch, left.port, left.position, left.time) != (right.batch, right.port, right.position, right.time):
            raise AssertionError("task input coordinates differ after RNG restore")
        torch.testing.assert_close(left.value, right.value, atol=0, rtol=0)
    return dict(schema="consumer-core5-task-check-v1", state="passed", device=str(actual_device),
                checks=["shared input root", "independent weighted head/loss formula", "input and head gradients", "exact data RNG/position continuation"],
                not_checked=["graph execution", "unified optimizer", "graph/task disk checkpoint", "other devices"],
                next="tidegraph check the generated regression config --scope related, then run and resume in separate processes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="Explicit cpu, cuda:N or npu:N; no backend fallback")
    args = parser.parse_args()
    print(json.dumps(check(args.device), indent=2))


if __name__ == "__main__":
    main()
