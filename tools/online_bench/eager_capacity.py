"""Shape/lifetime envelope for finite eager Add/Attention consumer runs.

Estimates require allocator calibration. They are not a vendor allocation proof,
nor do declared operator budgets count as allocated memory. Only independent
sample slices may shrink; model, dtype, position counts and update boundaries stay.
"""
from .eager_placement import owner_indices
from .traffic_bounds import traffic_bounds, checked

MIB=1024**2


def envelope(packet, owners, elements, traffic, *, physical_rows, payload, positions,
             connected_positions, training, adamw, workers, host_rss=False):
    g,c=packet["graph"],packet["workload"]
    width,vocab,batch=c["width"],c["vocab"],c["batch"]
    attention=c["memory"]=="attention"
    records=[]
    for device in range(len(elements)):
        nodes=[i for i in range(g["nodes"]) if owners[i]==device]
        atoms=traffic["owner_body_atom_factors"][device]
        events=traffic["owner_body_event_factors"][device]
        emissions=traffic["owner_body_emission_factors"][device]
        max_events=max([traffic["node_event_factors"][i] for i in nodes]+[0])*connected_positions
        max_keys=max([traffic["node_atom_factors"][i] for i in nodes]+[0])*positions
        learned=payload*elements[device]
        # Fixed scaffold tensors are cached by shape/kind/device. Tensor/record
        # metadata is conservatively charged even on HBM though it lives on host.
        constants=payload*(2*width*width+16*width+16)+4096*len(nodes)
        persistent=batch*(len(nodes)*(payload*width+4096))
        cache=atoms*positions*(2*width+1)*payload if attention else 0
        persistent+=batch*cache
        masters=2*learned if training and payload==2 else 0
        # Half leaves retain payload gradients plus FP32 master gradients/slots.
        optimizer=((7 if adamw else 5) if payload==2 else (3 if adamw else 2))*learned if training else 0
        vector_work=physical_rows*connected_positions*(
            payload*width*(64*atoms+32*events+32*emissions)+8192*(atoms+events+emissions))
        cache_work=score_work=0
        if attention:
            # Packed values, scalar semantic replay, immutable cache proposals
            # and cotangent work can coexist. Bound cumulative prefixes by the
            # largest per-node action count times the per-owner key inventory.
            cache_work=physical_rows*cache*(4*(max_events+2) if training else 4)
            score_work=physical_rows*atoms*connected_positions*max_keys*4*16
        largest_body=(3 if attention else 1)*width*width
        largest=max(largest_body,width*vocab if device==0 else 0)
        # Independent node workers may overlap local GEMM/gradient workspaces.
        operator_work=8*payload*largest_body*workers+4*payload*largest+64*MIB
        head_work=0
        if device==0:
            rows=physical_rows*connected_positions*traffic["output_frame_factor"]
            head_work=(8*payload*rows*vocab+8*payload*rows*width+3*payload*vocab*width
                       if training else 4*payload*rows*vocab+2*payload*rows*width)
            head_work+=64*MIB
            persistent+=batch*payload*width*2+batch*8192
        transport=16*MIB  # one bounded source/destination pack; payload charged above
        backend=512*MIB
        # Fresh original-width CPU observations include retained allocation
        # buffers beyond live tensor bytes (up to 4.3% of learned storage at
        # construction). Charge 6.25% at every phase; accelerator allocated-byte
        # accounting is separate and does not inherit this RSS-only allowance.
        rss_allowance=(learned+masters+15)//16 if host_rss else 0
        base=learned+masters+constants+persistent+backend+rss_allowance
        # The named CPU initializer can hold three int64 intermediates before
        # conversion. Eight payload rows cover that peak for supported training
        # precisions, as well as the final tensor/conversion transient.
        phases=dict(construction=learned+masters+constants+backend+rss_allowance+max(32,8*payload)*largest,
            forward=base+optimizer+vector_work+cache_work+score_work+operator_work+head_work+transport,
            backward=base+optimizer+vector_work+cache_work+score_work+operator_work+head_work+transport if training else 0,
            optimizer=base+optimizer+operator_work if training else 0)
        components=dict(learned=learned,master_parameters=masters,constants=constants,persistent_state_and_kv=persistent,
            gradients_and_optimizer_slots=optimizer,vector_work=vector_work,cache_work=cache_work,
            attention_scores=score_work,operator_work=operator_work,head_work=head_work,
            transport=transport,backend_allowance=backend,host_rss_allowance=rss_allowance)
        for value in [*components.values(),*phases.values()]:checked(value)
        records.append(dict(logical_device=device,components=components,phases=phases,
                            estimated_peak_bytes=max(phases.values())))
    return records


def plan(packet, *, budgets, dtype="float32", training=False, optimizer="sgd", steps=3, warmup=1,
         windows=2, workers=1, sample_rows=0, auto_sample_chunks=False, policy="conservative",
         owner_policy="locality", owner_map=(), head_workspace_bytes=4*1024**3, backend="cpu"):
    if dtype not in ("float16","float32","float64"):
        raise ValueError("eager capacity requires FP16/FP32/FP64 payload")
    if (not budgets or any(type(x) is not int or not 0<x<2**63 for x in budgets)
            or any(type(x) is not int or not 1<=x<2**63 for x in (steps,windows,workers,head_workspace_bytes))
            or type(warmup) is not int or not 0<=warmup<2**63
            or type(sample_rows) is not int or not 0<=sample_rows<2**63
            or type(auto_sample_chunks) is not bool or type(training) is not bool
            or optimizer not in ("sgd","adamw") or policy not in ("conservative","aggressive")
            or backend not in ("cpu","cuda","npu")):
        raise ValueError("invalid eager capacity geometry/budget")
    owners,elements=owner_indices(packet,len(budgets),owner_policy,owner_map)
    traffic=traffic_bounds(packet,owners)
    c=packet["workload"];positions=checked((steps+warmup)*windows*c["tokens"])
    connected=checked(windows*c["tokens"])
    rows=min(sample_rows or c["batch"],c["batch"]);requested=rows;attempts=[]
    usable=[max(0,v-v//(10 if policy=="aggressive" else 4)-128*MIB) for v in budgets]
    head_usable=head_workspace_bytes-head_workspace_bytes//(10 if policy=="aggressive" else 4)
    while True:
        devices=envelope(packet,owners,elements,traffic,physical_rows=rows,
            payload={"float16":2,"float32":4,"float64":8}[dtype],positions=positions,
            connected_positions=connected,training=training,adamw=optimizer=="adamw",workers=workers,
            host_rss=backend=="cpu")
        head=devices[0]["components"]["head_work"]
        accepted=all(d["estimated_peak_bytes"]<=limit for d,limit in zip(devices,usable)) and head<=head_usable
        attempts.append(dict(physical_rows=rows,estimated_peak_bytes=[d["estimated_peak_bytes"] for d in devices],head_bytes=head,accepted=accepted))
        if accepted or not auto_sample_chunks or rows==1:
            break
        rows=max(1,rows//2)
    for d,budget,limit in zip(devices,budgets,usable):
        d.update(budget_bytes=budget,usable_bytes=limit)
    return dict(schema="tide-eager-capacity-v1",state="admitted" if accepted else "refused",
        scope="static finite-run envelope for declared eager consumers; allocator calibration required",
        observation_counter="host_rss" if backend=="cpu" else "framework_allocator",
        requested_sample_rows=requested,effective_sample_rows=rows,logical_batch=c["batch"],
        physical_chunks=(c["batch"]+rows-1)//rows,sample_reductions=len(attempts)-1,
        attempts=attempts,devices=devices,head_workspace_bytes=head_workspace_bytes,
        head_usable_bytes=head_usable,chunk_policy=policy,node_owners=owners,
        parameter_elements=elements,traffic=traffic,positions=positions,
        connected_positions=connected)
