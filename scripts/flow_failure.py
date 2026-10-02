"""A non-successful consumer result whose diagnostic measurements must survive."""


class RecordedFailure(RuntimeError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = dict(record, state="failed", error=message)
