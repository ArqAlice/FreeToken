"""Per-request MTP depth selection from completed-cycle timings."""

from dataclasses import dataclass, field


@dataclass
class MTPPolicy:
    max_depth: int = 3
    min_samples: int = 4
    warmup_samples: int = 16
    depth: int = 0
    stopped: bool = False
    bucket: int | None = None
    costs: dict = field(default_factory=dict)
    samples: dict = field(default_factory=dict)
    probing: bool = True
    updates: int = 0
    warmup: int = 0
    baseline_age: int = 0
    elapsed: dict = field(default_factory=dict)
    emitted: dict = field(default_factory=dict)

    def context(self, position):
        bucket = position // 8192
        if self.bucket != bucket and not self.stopped:
            self.bucket = bucket
            self.costs.clear()
            self.samples.clear()
            self.elapsed.clear()
            self.emitted.clear()
            self.depth, self.probing, self.updates = 0, True, 0
            self.warmup, self.baseline_age = 0, 0

    def observe(self, depth, seconds, emitted):
        if self.stopped or emitted <= 0 or seconds <= 0:
            return
        if depth == 0 and self.warmup < self.warmup_samples:
            self.warmup += 1
            return
        # A cycle emitting several tokens must contribute their full weight.
        self.elapsed[depth] = .75 * self.elapsed.get(depth, seconds) + .25 * seconds
        self.emitted[depth] = .75 * self.emitted.get(depth, emitted) + .25 * emitted
        self.costs[depth] = self.elapsed[depth] / self.emitted[depth]
        self.samples[depth] = self.samples.get(depth, 0) + 1
        if depth != self.depth or self.samples[depth] < self.min_samples:
            return
        if depth == 0:
            if self.probing:
                self.depth = 1
            else:
                self._choose()
            return
        if self.probing:
            if depth == 1 and self.costs[1] >= self.costs[0] * .97:
                self.depth, self.stopped = 0, True
            elif depth < self.max_depth:
                self.depth += 1
            else:
                self.probing = False
                self._choose()
        else:
            self.updates += 1
            self.baseline_age += 1
            if self.updates >= 16:
                self.updates = 0
                self._choose()
            if not self.stopped and self.baseline_age >= 32:
                self.depth, self.baseline_age = 0, 0
                self.samples[0] = 0

    def _choose(self):
        best = min(self.costs, key=self.costs.get)
        if self.costs[best] >= self.costs[0] * .97:
            self.depth, self.stopped = 0, True
        elif (self.depth == 0 or self.costs[self.depth] >= self.costs[0] * .97
              or self.costs[best] < self.costs[self.depth] * .9):
            self.depth = best
            self.stopped = best == 0
