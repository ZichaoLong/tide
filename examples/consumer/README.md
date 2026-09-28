# Independent experiment starter

Copy this directory outside the Tide checkout. In an environment that already
has the intended Torch stack, install a pinned Tide wheel with
`python -m pip install --no-deps /path/to/tidegraph-0.2.0-py3-none-any.whl`.

```sh
python train.py --device cpu --output-dir results/python
python train.py --device cpu --implementation native \
  --native-library /path/to/matching/build --output-dir results/native
python -m tidegraph qualify graph.json --device cpu --output-dir results/gate
```

`train.py` owns a tiny regression dataset, head, loss, optimizer and output
directory. It trains three updates and runs inference. It is a consumption
example, not a convergence experiment. Replace those application pieces in the
new repository. The library checkout is never a runtime output directory.

Pin the library version/commit and retain `runtime.manifest()` with experiment
records. Graph checkpoints and task checkpoints are separate here; exact whole
application resume requires an application-owned atomic bundle and its tests.
See Tide's `docs/library.md` for family, config and checkpoint contracts.
