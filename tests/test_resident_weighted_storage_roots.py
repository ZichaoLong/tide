"""Public raw roots against literal KV matrices and the RMS Jacobian."""
import pytest
import torch
from resident_test_target import target_device
from test_resident_weighted_training import owner
from test_weighted_storage_reference import fixture, literal


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("name", ["keys", "summaries", "weights"])
def test_public_raw_field_has_independent_literal_vjp(name, storage_dtype):
    device = target_device(); b, xs = fixture(storage_dtype)
    expected, bar, input_bar, matrix_bar, parameter = literal(b, xs, name)
    with torch.no_grad():
        session = owner(b, device)
        w = session.advance(xs, stop=3, sealed_until=3)
        key = "node.0.final."+name; field = w.fields[key]
        assert field.values.dtype == storage_dtype and field.present.tolist() == [True]
        torch.testing.assert_close(field.values[0, 0].cpu(), expected.to(storage_dtype), rtol=1e-5, atol=1e-6)
        root = torch.zeros_like(field.values, dtype=torch.float32); root[0, 0] = bar.to(device)
        gradients = session.backward([session.cotangents(w, {key: root})])
        boundary = gradients.boundaries[0]
        rows = boundary.valid.nonzero().flatten()
        assert len(rows) == 1 and boundary.connected[rows[0]]
        assert boundary.values.dtype == torch.float32
        torch.testing.assert_close(boundary.values[rows[0]].cpu(), input_bar, rtol=1e-5, atol=1e-6)
        found = False
        for p, offset, connected in zip(gradients.parameters.names, gradients.parameters.offsets,
                                        gradients.parameters.connected.cpu().tolist()):
            if parameter and p.endswith("kernel."+parameter):
                found = True; assert connected
                torch.testing.assert_close(gradients.parameters.values[offset:offset+matrix_bar.numel()].cpu().reshape(matrix_bar.shape),
                                           matrix_bar, rtol=1e-5, atol=1e-6)
            if "read_program" in p or "full_program" in p or p.startswith("regions."):
                assert not connected, p
        assert found == (parameter is not None)
        session.detach(); session.close()
