"""NPU runtime boundary for the complete consumer's static capacity plan."""
from dataclasses import replace
import torch
from .capacity import Capacities, Chunks, packet_geometry, plan_samples


def prepare(packet, device, forward, limits, owners, head, training, optimizer, windows, budget, payload, diagnostics,
            sample_rows=None, context_memory_bytes=0, auto_sample_chunks=False):
    if type(budget) is not int or not 0 <= budget < 2**63:
        raise ValueError('device-memory-bytes must be a nonnegative int64')
    devices = list(owners.devices) or [str(device)]
    samples = []
    for value in devices:
        d = torch.device(value)
        if d.type != 'npu' or d.index is None:
            raise ValueError('consumer memory admission needs explicit logical NPUs')
        with torch.npu.device(d):
            free,total = torch.npu.mem_get_info(d)
            samples.append(dict(device=str(d),free_bytes=free,total_bytes=total,
                                allocated_bytes=torch.npu.memory_allocated(d)))
    g = packet_geometry(packet,windows=windows,payload=payload,
        training=training,adamw=optimizer=='adamw',diagnostics=diagnostics,
        devices=len(devices),locality=owners.policy=='locality')
    if sample_rows is not None:
        g = replace(g,batch=sample_rows,sample_chunks=(g.batch-1)//sample_rows+1)
    g = replace(g,context_bytes=context_memory_bytes)
    c = Capacities(forward.queue,forward.arrivals,forward.outputs,forward.trace,forward.kv_rows,forward.kv_trace_rows,limits.program_workspace_bytes)
    chunks = Chunks(forward.full_chunk_rows,forward.emission_chunk_rows,forward.aggregate_chunk_rows,
                    forward.attention_chunk_rows,forward.attention_key_rows,limits.reverse_chunk_rows,head.rows)
    record = plan_samples(g,c,chunks,[min(s['free_bytes'],budget) if budget else s['free_bytes'] for s in samples],
                  forward.chunk_policy=='aggressive',packet['workload']['batch'],auto_sample_chunks,owners.full_owners,owners.state_owners)
    record['requested_device_memory_bytes'] = budget
    record['initial_devices'] = samples
    selected = record['effective_chunks']
    forward = replace(forward,full_chunk_rows=selected['full'],emission_chunk_rows=selected['emission'],
                      aggregate_chunk_rows=selected['aggregate'],attention_chunk_rows=selected['attention'],attention_key_rows=selected['keys'])
    limits = replace(limits,reverse_chunk_rows=selected['reverse'])
    head = replace(head,rows=selected['head'],reserved_bytes=head.fixed_bytes+selected['head']*head.row_bytes)
    if owners.devices:
        owners = replace(owners,full_owners=tuple(record['full_owners']),state_owners=tuple(record['state_owners']))
    return forward,limits,owners,head,record


def observed(record, memory):
    if len(record['devices']) != len(record['initial_devices']):
        raise ValueError('consumer memory observation device mismatch')
    observed = []
    for card,initial in zip(record['devices'],record['initial_devices']):
        peak = max(d['peak_allocated_bytes'] for phase in memory['phases'] for d in phase['devices'] if d['device']==initial['device'])
        observed.append(max(0,peak-initial['allocated_bytes']))
    record['observed_peak_growth_bytes'] = observed
    record['allocator_within_estimate'] = all(value <= card['estimated_peak_bytes']
        for value,card in zip(observed,record['devices']))
    return record['allocator_within_estimate']
