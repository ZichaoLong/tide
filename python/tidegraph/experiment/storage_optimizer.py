"""Independent reference SGD: FP32 arithmetic/master and half parameter stores.

The numerical model contains widened half values. Its ordinary CPU autograd
produces FP32 gradients. This adapter uses PyTorch SGD, never resident updates.
"""
import copy
import torch


class StorageSGD:
    def __init__(self, parameters, *, lr=.001, momentum=.3, weight_decay=.01):
        self.payload = list(parameters)
        self.masters = [p.detach().clone().requires_grad_() for p in self.payload]
        self.inner = torch.optim.SGD(self.masters, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def zero_grad(self, set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)
        for p in self.payload:
            p.grad = None

    def backward(self, loss):
        loss.backward()

    @torch.no_grad()
    def step(self):
        for p, master in zip(self.payload, self.masters):
            master.grad = p.grad
        self.inner.step()
        for p, master in zip(self.payload, self.masters):
            p.copy_(master.half().float())

    def state_dict(self):
        state = copy.deepcopy(self.inner.state_dict())
        state["precision"] = dict(schema="tide-fp32-masters-v1", optimizer="sgd", loss_scale=1.,
                                  masters=[p.detach().clone() for p in self.masters])
        return state
