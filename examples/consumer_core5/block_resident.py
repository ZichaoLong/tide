"""Installed public block client: independent Python schedule, training, restore.

Run in a fresh process with a matching native/plugin build. This imports the
installed tidegraph package; it never prepends this checkout's Python sources.
"""
import argparse
from pathlib import Path
import torch
import tidegraph
from tidegraph import Continuation
from tidegraph.compare import equivalent
from tidegraph.reference import run as reference
from tidegraph.weighted import (KVConfig, RegionalConfig, SourceBinding, bind_layered,
                               fully_connected, WeightedResidentTraining, WeightedBlockResidentTraining)
from block_observations import reference_prefixes, prefix_loss, check_prefixes


def fixture(memory_read="qry"):
    return bind_layered(fully_connected(2, 2),
        memory=KVConfig(width=2, summary_width=2, adapter_rank=1, ffn_width=3,
                        memory_read=memory_read, epsilon=.3, full_frequencies=((.23,),),
                        read_frequencies=(.17,) if memory_read == "qry" else ()),
        regions=RegionalConfig(k=2, gamma_max=1.6, hidden_width=3),
        window_positions=4, alpha_agg=.3, source_hidden_width=3,
        source_binding=SourceBinding("adapted", 1), dtype=torch.float32)


def exact(a, b):
    assert (a.schema, a.identity, a.generation, a.next_token, a.aliases) == (b.schema, b.identity, b.generation, b.next_token, b.aliases)
    for name, x in a.parameters.items():
        assert torch.equal(x, b.parameters[name]), name
    for name in ("last_positions", "node_present", "region_times", "input_counts", "input_times"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    for name in ("values", "first", "second", "maximum", "steps", "corrections"):
        assert torch.equal(getattr(a.slots, name), getattr(b.slots, name)), name
    for group in ("state", "bound_initial"):
        for x, y in zip(getattr(a, group), getattr(b, group), strict=True):
            assert torch.equal(x.floating, y.floating) and torch.equal(x.integer, y.integer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-library", required=True)
    parser.add_argument("--resident-library", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--restore-check", action="store_true")
    parser.add_argument("--memory-read", choices=("qry", "sum"), default="qry")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if Path(tidegraph.__file__).resolve().is_relative_to(root/"python"):
        raise RuntimeError("public installation check must use an independently installed tidegraph")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec, binding = fixture(args.memory_read)
    _, oracle = fixture(args.memory_read)
    parameters = [p for p in oracle.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.003, eps=.1, weight_decay=.02, foreach=False)
    initial = Continuation(binding.graph.identity, 2)
    q = Continuation(oracle.graph.identity, 2)
    kwargs = dict(device=args.device, native_library=args.native_library, resident_library=args.resident_library,
                  batch_size=2, limits=dict(cache=16, forward_bytes=1 << 29, backward_bytes=1 << 30, total_bytes=3 << 30))
    with torch.no_grad():
        if args.restore_check:
            session = WeightedResidentTraining(spec, binding.model, checkpoint=args.checkpoint_path, **kwargs)
        else:
            # Explicit canonical group names belong to the native parameter
            # registry; the default group is then changed only in its options.
            from tidegraph.native_loader import load_native
            from tidegraph.native_model import encode_model
            core = load_native(args.native_library, backend=args.device.split(":", 1)[0])
            _, native_model = encode_model(core, binding.graph, binding.model)
            groups = [dict(parameters=[p.canonical for p in native_model.parameters().owners()], lr=.003, eps=.1, weight_decay=.02)]
            session = WeightedResidentTraining(spec, binding.model, optimizer="adamw", groups=groups, **kwargs)
        assert isinstance(session, WeightedBlockResidentTraining)
    aliased = dict(oracle.model.named_parameters(remove_duplicate=False))
    for step in range(3):
        positions = torch.arange(3*step, 3*step+3, dtype=torch.int64).repeat(2, 1)
        payload = torch.full((2, 3, 2, 3), .12)
        payload[..., 0] = .3
        payload[0, 1, 1, 0] = 0
        present = torch.ones((2, 3, 2), dtype=torch.bool)
        present[1, 1, 0] = False
        inputs, leaves, rows = [], [], []
        for b in range(2):
            for p in range(2):
                count = q.ledger.get((b, p), (-1, -1))[0]+1
                for l in range(3):
                    if not present[b, l, p]:
                        continue
                    value = payload[b, l, p].clone().requires_grad_()
                    inputs.extend(spec.port_external(value[None], [int(positions[b, l])], port=p, batch=b, start_occurrence=count))
                    leaves.append(value); rows.append((b*3+l)*2+p); count += 1
        stop = (step+1)*3*spec.stride
        result = reference(oracle.graph, oracle.model, q, inputs, stop, sealed_until=stop)
        loss = sum(x[0]*.07+x[1:].sum()*.13 for *_, x in result.outputs)
        prefixes = reference_prefixes(result, q, oracle.model)
        loss = loss+prefix_loss(prefixes)
        expected = torch.autograd.grad(loss, [*parameters, *leaves], allow_unused=True)
        if not (args.restore_check and step == 0):
            with torch.no_grad():
                window = session.advance(positions, payload, present)
                expected_output = {(b, t, p): x for b, t, p, x in result.outputs}
                mass, content = window.fields["output.strength"], window.fields["output.content"]
                for b in range(2):
                    for l in range(3):
                        for p, node in enumerate(spec.graph.outputs):
                            key = b, int(positions[b, l])*spec.stride+spec.rank(node), p
                            row = b*3+l
                            assert bool(mass.present[row, p]) == (key in expected_output)
                            if key in expected_output:
                                torch.testing.assert_close(mass.values[row, p].cpu(), expected_output[key][0], rtol=8e-5, atol=8e-6)
                                torch.testing.assert_close(content.values[row, p].cpu(), expected_output[key][1:], rtol=8e-5, atol=8e-6)
                raw_roots = check_prefixes(session, window, spec, prefixes)
                roots = session.cotangents(window, {"output.strength": torch.full_like(mass.values, .07),
                    "output.content": torch.full_like(content.values, .13), **raw_roots})
                actual = session.backward([roots])
                expected_by_id = {id(p): g for p, g in zip(parameters, expected, strict=False)}
                for j, aliases in enumerate(actual.parameters.aliases):
                    parameter = aliased.get(aliases[0].replace(".extra.", "."))
                    if parameter is None or not parameter.requires_grad:
                        continue
                    gradient = expected_by_id[id(parameter)]
                    assert bool(actual.parameters.connected[j]) == (gradient is not None), aliases[0]
                    if gradient is not None:
                        value = actual.parameters.values.narrow(0, actual.parameters.offsets[j], parameter.numel()).reshape_as(parameter).cpu()
                        torch.testing.assert_close(value, gradient, rtol=8e-5, atol=8e-6, msg=aliases[0])
                for index, row in enumerate(rows):
                    gradient = expected[len(parameters)+index]
                    assert bool(actual.inputs[0].connected.reshape(-1)[row]) == (gradient is not None)
                    if gradient is not None:
                        torch.testing.assert_close(actual.inputs[0].values.reshape(-1, 3)[row].cpu(), gradient, rtol=8e-5, atol=8e-6)
                session.step()
        for parameter, gradient in zip(parameters, expected, strict=False):
            parameter.grad = gradient
        optimizer.step(); optimizer.zero_grad(set_to_none=True); q = result.continuation.detach()
        if args.restore_check and step == 0:
            continue
        with torch.no_grad():
            equivalent(session.continuation(), q, rtol=8e-5, atol=8e-6)
            checkpoint = session.checkpoint()
            for name, value in checkpoint.parameters.items():
                parameter = aliased.get(name.replace(".extra.", "."))
                if parameter is not None:
                    torch.testing.assert_close(value, parameter, rtol=8e-5, atol=8e-6, msg=name)
            if step == 0:
                session.save(args.checkpoint_path)
    with torch.no_grad():
        if args.restore_check:
            exact(session.checkpoint(), session.module.load_block_checkpoint(str(args.checkpoint_path)+".expected"))
        else:
            session.save(str(args.checkpoint_path)+".expected")
        session.close()
    print(dict(state="passed", default="settle-block-v2", source=str(tidegraph.__file__),
               memory_read=args.memory_read, independent_python_schedule=True,
               explicit_raw_observation=True, independent_proposal_roots=True,
               adamw_steps=3, fresh_process=args.restore_check))


if __name__ == "__main__":
    main()
