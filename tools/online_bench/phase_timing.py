"""Optional coarse wall timing; callers synchronize at the optimizer boundary."""


class PhaseTiming:
    def __init__(self, enabled=False):
        if type(enabled) is not bool:
            raise ValueError("phase-timing must be boolean")
        self.enabled = enabled
        self.measured, self.warmup = [], []

    def add(self, total, sample, warming):
        if self.enabled:
            sample = total if sample is None else sample
            (self.warmup if warming else self.measured).append(
                dict(sample_work_seconds=sample, optimizer_seconds=total-sample))

    def record(self):
        return dict(enabled=self.enabled, measured=self.measured, warmup=self.warmup,
                    boundary="training: all resolved devices synchronized before optimizer; extra synchronization included in total",
                    scope="sample work includes input/forward/loss/backward/accumulation/continuation; optimizer includes final finite checks and publication; inference has zero optimizer time")
