"""Architecture §2.2 / FC §6.2: exact Aggregate computation-sharing identity.

Equal numeric values of independent Tensor roots are deliberately distinct.
The scheduler retains every atom, event and private state. This key promises
equal complete Aggregate results for identical *ordered* physical value/class
rows and parameter handles; time and local port IDs are not formula inputs.
"""


def aggregate_key(program, request):
    rows = []
    for source in request.sources:
        label = program.classes.classify(source.atom)
        tensors = ()
        if program.alpha:
            transform = program.transforms[str(label)]
            tensors = tuple((name, id(p)) for name, p in sorted(transform.named_parameters()))
        rows.append((label, id(source.atom.value), tensors))
    return program.alpha, program.epsilon, tuple(rows)
