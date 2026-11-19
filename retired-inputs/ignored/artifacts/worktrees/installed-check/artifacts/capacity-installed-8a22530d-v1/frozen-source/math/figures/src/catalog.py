"""Figure registry and definition paths, relative to the figures directory."""

REGISTRY = {}
ARCH = '../local-semantics/weighted-message/architecture-learning-note.md'
ALLOC = '../local-semantics/weighted-message/region-allocation-learning-note.md'
KV = '../local-semantics/weighted-message/kv-memory-learning-note.md'
REC = '../local-semantics/weighted-message/recurrent-memory/'


def register(name, title, sources, purpose, assumptions=()):
    def decorate(function):
        REGISTRY[name] = dict(title=title, sources=sources, purpose=purpose,
                              assumptions=list(assumptions), draw=function)
        return function
    return decorate
