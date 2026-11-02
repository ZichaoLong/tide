"""Fixed first-order LM recipe; token-weighted accumulation and complete update timing."""
from contextlib import nullcontext
import time
from .runtime import synchronize

RECIPE = dict(optimizer='AdamW', lr=1e-4, betas=[0.9, 0.999], eps=1e-8, weight_decay=0,
              foreach=False, loss='shifted nonpadding cross entropy mean', olmoe_aux_loss_coefficient=0.01,
              full='FP32 parameters/gradients/slots; configured autocast compute',
              lora='frozen base/router; Q/V r8 alpha16 dropout0; FP32 adapters/slots')


def optimizer_for(owner):
    import torch
    return torch.optim.AdamW([p for p in owner.parameters() if p.requires_grad],
                            lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0., foreach=False)


def update(owner, optimizer, host_ids, host_mask, c, device, backend, *, observer=None):
    import torch
    t = c['task']; e = c['execution']
    synchronize(backend, e['devices']); begin = time.perf_counter()
    ids = host_ids.to(device); mask = host_mask.to(device)
    labels = ids.clone(); labels.masked_fill_(mask == 0, -100)
    optimizer.zero_grad(set_to_none=True)
    micro = t['microbatch']; parts = t['global_batch']//micro
    losses = []; aux = []; all_finite = torch.ones((), dtype=torch.bool, device=device)
    for start in range(0, t['global_batch'], micro):
        context = nullcontext() if e['dtype'] == 'float32' else torch.autocast(
            device_type=e['backend'], dtype=getattr(torch, e['dtype']))
        with context:
            result = owner(input_ids=ids[start:start+micro], attention_mask=mask[start:start+micro],
                           labels=labels[start:start+micro], use_cache=False,
                           **({'output_router_logits': True} if c['model'] == 'olmoe' else {}))
            loss = result.loss.float()/parts
        all_finite &= torch.isfinite(loss)
        loss.backward(); losses.append(loss.detach())
        if getattr(result, 'aux_loss', None) is not None:
            aux.append(result.aux_loss.detach().float()/parts)
    gradients = dict(present=0, none=0, elements=0)
    for p in owner.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            gradients['none'] += 1
        else:
            all_finite &= torch.isfinite(p.grad).all()
            gradients['present'] += 1; gradients['elements'] += p.numel()
    # Check before update: an invalid gradient may never be accepted as an exploratory step.
    if not bool(all_finite.cpu()):
        raise FloatingPointError('nonfinite loss or gradient; optimizer not advanced')
    if observer:observer("gradient",owner,optimizer)
    optimizer.step()
    for p in owner.parameters():
        if p.requires_grad:
            all_finite &= torch.isfinite(p).all()
    for slots in optimizer.state.values():
        for value in slots.values():
            if isinstance(value, torch.Tensor):
                all_finite &= torch.isfinite(value).all().to(device)
    values = torch.stack([sum(losses), sum(aux) if aux else losses[0].new_zeros(())]).cpu().tolist()
    if not bool(all_finite.cpu()):
        raise FloatingPointError('nonfinite updated parameter or optimizer slot')
    if observer:observer("updated",owner,optimizer)
    optimizer.zero_grad(set_to_none=True)
    synchronize(backend, e['devices']); elapsed = time.perf_counter()-begin
    tokens = t['global_batch']*(t['input_tokens']-1)
    return dict(total_seconds=elapsed, loss=values[0], auxiliary_loss=values[1],
                target_tokens=tokens, target_tokens_per_s=tokens/elapsed,
                global_batch=t['global_batch'], microbatch=micro, accumulation_steps=parts,
                gradients=gradients, optimizer_steps=1)


def train(owner, host_ids, host_mask, c, out, device, backend):
    from ..experiment.records import write_json
    opt = optimizer_for(owner)
    warm = [update(owner, opt, host_ids, host_mask, c, device, backend) for _ in range(c['measurement']['warmup'])]
    measured = []
    for index in range(c['measurement']['steps']):
        measured.append(update(owner, opt, host_ids, host_mask, c, device, backend))
        write_json(out/'progress.json', dict(completed_updates=index+1, measurements=measured))
    checkpoint = None
    if c['measurement']['checkpoint']:
        from .checkpoint import prepare_training
        checkpoint = prepare_training(owner, opt, host_ids, host_mask, c, out, device, backend)
    return dict(recipe=RECIPE, warmup=warm, measurements=measured, checkpoint_request=checkpoint,
                work=dict(logical_batch=c['task']['global_batch'], physical_microbatch=c['task']['microbatch'],
                          return_contract='loss/finite/gradient counts; gradients cleared after each update',
                          target_tokens_per_update=c['task']['global_batch']*(c['task']['input_tokens']-1)))
