"""Packed source arithmetic with separate summary/contribution gradient roots."""
import torch


class _Sources(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sources, mean, coefficients, *inputs):
        ctx.set_materialize_grads(False)
        count = len(inputs)//2
        atoms, scales = inputs[:count], inputs[count:]
        rows = count//sources
        x = torch.stack(atoms).reshape(rows, sources, -1)
        s = torch.stack(scales).reshape(rows, sources)
        terms = x*s.unsqueeze(-1)
        if mean:
            terms = terms/sources
        elif coefficients.numel():
            terms = terms*coefficients.reshape(-1, sources, 1)
        total = terms[:, 0]
        for j in range(1, sources):
            total = total + terms[:, j]
        outputs, frozen = [], []
        for i in range(rows):
            summary = total[i]
            outputs.append(summary)
            active = False
            for j in range(sources):
                k = i*sources+j
                value = terms[i, j]
                outputs.append(value)
                needs = atoms[k].requires_grad or scales[k].requires_grad or coefficients.requires_grad
                active |= needs
                if not needs:
                    frozen.append(value)
            if not active:
                frozen.append(summary)
        ctx.mark_non_differentiable(*frozen)
        ctx.sources, ctx.mean = sources, mean
        ctx.save_for_backward(x, s, coefficients)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *bars):
        if torch.is_grad_enabled():
            raise ValueError("isolated Aggregate supports first-order VJP only")
        sources = ctx.sources
        rows = len(bars)//(sources+1)
        count = rows*sources
        result = [None]*(3+2*count)
        x, s, coe = ctx.saved_tensors
        dc = None
        for j in range(sources):
            used, values = [], []
            for i in range(rows):
                total, part = bars[i*(sources+1)], bars[i*(sources+1)+1+j]
                if total is None and part is None:
                    continue
                used.append(i)
                values.append(part if total is None else total if part is None else total+part)
            if not used:
                continue
            dy = torch.stack(values)
            xs, ss = x[used, j], s[used, j]
            dv = dy/sources if ctx.mean else dy
            if not ctx.mean and coe.numel():
                dv = dy*(coe[used, j].unsqueeze(-1) if coe.ndim == 2 else coe[j])
            if any(ctx.needs_input_grad[3+i*sources+j] for i in used):
                dx = dv*ss.unsqueeze(-1)
                for k, i in enumerate(used):
                    if ctx.needs_input_grad[3+i*sources+j]:
                        result[3+i*sources+j] = dx[k]
            if any(ctx.needs_input_grad[3+count+i*sources+j] for i in used):
                ds = (dv*xs).sum(-1)
                for k, i in enumerate(used):
                    if ctx.needs_input_grad[3+count+i*sources+j]:
                        result[3+count+i*sources+j] = ds[k]
            if coe.numel() and ctx.needs_input_grad[2]:
                if dc is None:
                    dc = torch.zeros_like(coe)
                terms = dy*(xs*ss.unsqueeze(-1))
                if coe.ndim == 2:
                    dc[used, j] = terms.sum(-1)
                else:
                    dc[j] = terms.sum()
        result[2] = dc
        return tuple(result)


def aggregate(atoms, scales, coefficients, sources, mean=False):
    if type(sources) is not int or sources < 1 or not atoms or len(atoms)%sources or len(atoms) != len(scales):
        raise ValueError("isolated Aggregate source layout mismatch")
    first = atoms[0]
    if first.ndim != 1 or first.device.type not in {"cpu", "cuda", "npu"} or first.dtype not in (torch.float32, torch.float64):
        raise ValueError("isolated Aggregate requires supported FP32/FP64 vectors")
    if any(a.shape != first.shape or s.ndim != 0 or a.dtype != first.dtype or s.dtype != first.dtype
           or a.device != first.device or s.device != first.device for a, s in zip(atoms, scales)):
        raise ValueError("isolated Aggregate input metadata mismatch")
    if (coefficients.shape not in {(0,), (sources,), (len(atoms)//sources, sources)}
            or coefficients.dtype != first.dtype or coefficients.device != first.device
            or (mean and coefficients.numel())):
        raise ValueError("isolated Aggregate coefficients mismatch")
    return list(_Sources.apply(sources, mean, coefficients, *atoms, *scales))
