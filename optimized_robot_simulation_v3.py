#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass
from typing import Deque, Dict, List, Optional, Sequence


EPS = 1e-9
STAGES: tuple[str, ...] = ("A", "B", "C", "E")
ABC_STAGES: tuple[str, ...] = ("A", "B", "C")


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


class QueuePolicy(StrEnum):
    FIFO = "fifo"
    PRIORITY = "priority"


class OperationKind(StrEnum):
    LOAD = "load"
    UNLOAD = "unload"
    PREDEBUG = "predebug"
    TEST = "test"


class EventKind(StrEnum):
    TASK_END = "task_end"
    REPAIR_DONE = "repair_done"


class TaskOutcome(StrEnum):
    COMPLETE = "complete"
    INTERRUPTED = "interrupted"
    EQUIPMENT_FAILURE = "equipment_failure"


@dataclass(frozen=True, slots=True)
class FailureBand:
    max_hour: float
    rate_per_hour: float


@dataclass(frozen=True, slots=True)
class Operation:
    kind: OperationKind
    stage: str | None = None

    @property
    def queue_stage(self) -> str | None:
        if self.kind in (OperationKind.PREDEBUG, OperationKind.TEST):
            return self.stage
        return None

    @property
    def label(self) -> str:
        if self.stage is None:
            return self.kind.value.upper()
        return f"{self.kind.value.upper()}_{self.stage}"


@dataclass(frozen=True, slots=True)
class StageConfig:
    name: str
    calibration_h: float
    test_h: float
    defect_rate: float
    operator_error_rate: float
    worker_count: int = 1
    station_count: int = 1
    queue_capacity: int | None = None
    queue_policy: QueuePolicy = QueuePolicy.FIFO
    failure_bands: tuple[FailureBand, ...] = ()


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    stages: Dict[str, StageConfig]
    alpha: float = 0.60
    beta: float = 0.40
    load_h: float = 0.5
    unload_h: float = 0.5
    predebug_h: float = 0.5
    predebug_reduce_e: float = 0.30
    benches: int = 2
    repair_h: float = 1.0
    max_attempts_per_stage: int = 2
    latent_defect_rate_d: float = 0.002


@dataclass(frozen=True, slots=True)
class Scenario:
    n_robots: int
    part: int
    shift_hours: float
    handover_h: float = 0.5
    maintenance_every_days: int = 7
    maintenance_h: float = 4.0
    enable_predebug: bool = False


@dataclass(slots=True)
class Robot:
    robot_id: int
    defects: Dict[str, bool]
    hidden_defects: set[str]
    stage_pass: Dict[str, bool] = field(default_factory=lambda: {stage: False for stage in STAGES})
    attempts: Dict[str, int] = field(default_factory=lambda: {stage: 0 for stage in STAGES})
    bench_id: int | None = None
    loaded: bool = False
    eliminated: bool = False
    done: bool = False
    finish_time: float | None = None
    predebug_done: bool = False
    active_operation: Operation | None = None
    queued_stage: str | None = None

    def next_stage(self) -> str | None:
        for stage in STAGES:
            if not self.stage_pass[stage]:
                return stage
        return None

    def passed_count_abc(self) -> int:
        return sum(1 for stage in ABC_STAGES if self.stage_pass[stage])

    def has_hidden_defect(self) -> bool:
        return bool(self.hidden_defects)

    def is_qualified(self) -> bool:
        return (not self.eliminated) and all(self.stage_pass.values())


# ---------------------------------------------------------------------------
# Event system
# ---------------------------------------------------------------------------


@dataclass(order=True, slots=True)
class Event:
    time: float
    priority: int
    seq: int
    kind: EventKind = field(compare=False)
    robot_id: int | None = field(compare=False, default=None)
    operation: Operation | None = field(compare=False, default=None)
    outcome: TaskOutcome | None = field(compare=False, default=None)
    run_h: float = field(compare=False, default=0.0)
    stage: str | None = field(compare=False, default=None)
    bench_id: int | None = field(compare=False, default=None)
    worker_id: int | None = field(compare=False, default=None)
    station_id: int | None = field(compare=False, default=None)


class EventQueue:
    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._seq = 0

    def push(
        self,
        *,
        time: float,
        priority: int,
        kind: EventKind,
        robot_id: int | None = None,
        operation: Operation | None = None,
        outcome: TaskOutcome | None = None,
        run_h: float = 0.0,
        stage: str | None = None,
        bench_id: int | None = None,
        worker_id: int | None = None,
        station_id: int | None = None,
    ) -> None:
        event = Event(
            time=time,
            priority=priority,
            seq=self._seq,
            kind=kind,
            robot_id=robot_id,
            operation=operation,
            outcome=outcome,
            run_h=run_h,
            stage=stage,
            bench_id=bench_id,
            worker_id=worker_id,
            station_id=station_id,
        )
        self._seq += 1
        heapq.heappush(self._heap, event)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def next_time(self) -> float:
        return self._heap[0].time

    def pop_current_batch(self, now: float) -> list[Event]:
        events: list[Event] = []
        while self._heap and abs(self._heap[0].time - now) < EPS:
            events.append(heapq.heappop(self._heap))
        return events


class Scheduler:
    def __init__(self) -> None:
        self.time = 0.0
        self.event_queue = EventQueue()

    def schedule(self, *, time: float, priority: int, kind: EventKind, **payload: object) -> None:
        self.event_queue.push(time=time, priority=priority, kind=kind, **payload)

    def has_events(self) -> bool:
        return bool(self.event_queue)

    def advance_to_next_event(self) -> float:
        self.time = self.event_queue.next_time()
        return self.time

    def pop_current_events(self) -> list[Event]:
        return self.event_queue.pop_current_batch(self.time)


# ---------------------------------------------------------------------------
# Resource system
# ---------------------------------------------------------------------------


class Resource(ABC):
    def __init__(self, resource_id: int, stage: str | None) -> None:
        self.resource_id = resource_id
        self.stage = stage
        self.busy = False
        self.current_robot_id: int | None = None
        self.current_operation: Operation | None = None
        self.productive_hours = 0.0
        self.down_until = 0.0

    @property
    @abstractmethod
    def kind(self) -> str:
        raise NotImplementedError

    def is_available(self, now: float) -> bool:
        return (not self.busy) and self.down_until <= now + EPS

    def assign(self, *, robot_id: int, operation: Operation) -> None:
        self.busy = True
        self.current_robot_id = robot_id
        self.current_operation = operation

    def release(self, productive_h: float) -> None:
        self.busy = False
        self.current_robot_id = None
        self.current_operation = None
        self.productive_hours += productive_h


class Bench(Resource):
    def __init__(self, resource_id: int) -> None:
        super().__init__(resource_id=resource_id, stage=None)
        self.occupied_robot_id: int | None = None
        self.occupied_since: float | None = None
        self.occupied_calendar_hours = 0.0

    @property
    def kind(self) -> str:
        return "bench"

    def can_accept_robot(self, now: float) -> bool:
        return self.occupied_robot_id is None and self.is_available(now)

    def occupy(self, *, robot_id: int, now: float) -> None:
        self.occupied_robot_id = robot_id
        self.occupied_since = now

    def release_robot(self, now: float) -> None:
        if self.occupied_robot_id is None:
            return
        if self.occupied_since is not None:
            self.occupied_calendar_hours += max(0.0, now - self.occupied_since)
        self.occupied_robot_id = None
        self.occupied_since = None


class Worker(Resource):
    @property
    def kind(self) -> str:
        return "worker"


class Station(Resource):
    def __init__(self, resource_id: int, stage: str, failure_bands: Sequence[FailureBand]) -> None:
        super().__init__(resource_id=resource_id, stage=stage)
        self.failure_bands = tuple(failure_bands)
        self.cumulative_hours = 0.0
        self.failures = 0

    @property
    def kind(self) -> str:
        return "station"

    def release(self, productive_h: float) -> None:
        super().release(productive_h)
        self.cumulative_hours += productive_h


@dataclass(order=True, slots=True)
class QueueRequest:
    primary_key: int
    secondary_key: int
    robot_id: int = field(compare=False)
    operation: Operation = field(compare=False)


class StageSystem:
    def __init__(self, config: StageConfig) -> None:
        self.config = config
        self.name = config.name
        self.workers = [Worker(resource_id=i, stage=config.name) for i in range(config.worker_count)]
        self.stations = [
            Station(resource_id=i, stage=config.name, failure_bands=config.failure_bands)
            for i in range(config.station_count)
        ]
        self.queue: list[QueueRequest] = []
        self._queued_robot_ids: set[int] = set()

    def has_available_capacity(self, now: float) -> bool:
        return self._select_station(now) is not None and self._select_worker(now) is not None

    def can_enqueue(self, robot_id: int) -> bool:
        if robot_id in self._queued_robot_ids:
            return False
        if self.config.queue_capacity is None:
            return True
        return len(self.queue) < self.config.queue_capacity

    def enqueue(self, request: QueueRequest) -> bool:
        if not self.can_enqueue(request.robot_id):
            return False
        heapq.heappush(self.queue, request)
        self._queued_robot_ids.add(request.robot_id)
        return True

    def pop_request(self) -> QueueRequest | None:
        if not self.queue:
            return None
        request = heapq.heappop(self.queue)
        self._queued_robot_ids.discard(request.robot_id)
        return request

    def select_pair(self, now: float) -> tuple[Station, Worker] | None:
        station = self._select_station(now)
        worker = self._select_worker(now)
        if station is None or worker is None:
            return None
        return station, worker

    def effective_hours(self) -> float:
        return sum(station.productive_hours for station in self.stations)

    def worker_hours(self) -> float:
        return sum(worker.productive_hours for worker in self.workers)

    def failures(self) -> int:
        return sum(station.failures for station in self.stations)

    def reset_maintenance(self) -> None:
        for station in self.stations:
            station.cumulative_hours = 0.0

    def _select_station(self, now: float) -> Station | None:
        available = [station for station in self.stations if station.is_available(now)]
        if not available:
            return None
        return min(available, key=lambda item: (item.cumulative_hours, item.resource_id))

    def _select_worker(self, now: float) -> Worker | None:
        available = [worker for worker in self.workers if worker.is_available(now)]
        if not available:
            return None
        return min(available, key=lambda item: (item.productive_hours, item.resource_id))


class FailureModel:
    def sample_failure_time(
        self,
        *,
        station: Station,
        duration_h: float,
        rng: random.Random,
    ) -> float | None:
        if duration_h <= EPS:
            return None

        elapsed = 0.0
        cumulative = station.cumulative_hours

        while elapsed < duration_h - EPS:
            band = self._band_for_hours(station.failure_bands, cumulative)
            if band is None or band.rate_per_hour <= EPS:
                return None

            remaining = duration_h - elapsed
            band_limit = band.max_hour - cumulative
            if not math.isfinite(band_limit):
                segment_h = remaining
            else:
                segment_h = min(remaining, max(band_limit, 0.0))

            if segment_h <= EPS:
                cumulative += EPS
                continue

            candidate = rng.expovariate(band.rate_per_hour)
            if candidate < segment_h - EPS:
                return elapsed + max(candidate, EPS)

            elapsed += segment_h
            cumulative += segment_h

        return None

    @staticmethod
    def _band_for_hours(bands: Sequence[FailureBand], cumulative_h: float) -> FailureBand | None:
        for band in bands:
            if cumulative_h < band.max_hour - EPS:
                return band
        return bands[-1] if bands else None


class WorkCalendar:
    def __init__(self, scenario: Scenario) -> None:
        if scenario.part not in (2, 3):
            raise ValueError("scenario.part must be 2 or 3")
        self.scenario = scenario

    def is_work_time(self, t: float) -> bool:
        active, _, _ = self._shift_state(t)
        if not active:
            return False
        if self.scenario.part == 3 and self.in_maintenance(t):
            return False
        return True

    def current_work_window_end(self, t: float) -> float | None:
        if not self.is_work_time(t):
            return None
        _, _, shift_end = self._shift_state(t)
        if self.scenario.part == 3:
            return min(shift_end, self.next_maintenance_start(t))
        return shift_end

    def next_work_start(self, t: float) -> float:
        probe = t
        for _ in range(100_000):
            if self.is_work_time(probe):
                return probe
            if self.scenario.part == 3 and self.in_maintenance(probe):
                end = self.current_maintenance_end(probe)
                if end is None:
                    raise RuntimeError("maintenance state is inconsistent")
                probe = end + EPS
                continue
            probe = self.next_shift_start(probe + EPS)
        raise RuntimeError("next_work_start did not converge")

    def total_work_hours(self, t_end: float) -> float:
        total = 0.0
        t = 0.0
        while t < t_end - EPS:
            t = self.next_work_start(t)
            if t >= t_end - EPS:
                break
            end = self.current_work_window_end(t)
            if end is None:
                t += EPS
                continue
            total += max(0.0, min(end, t_end) - t)
            t = end + EPS
        return total

    def in_maintenance(self, t: float) -> bool:
        return self._maintenance_window(t) is not None

    def current_maintenance_end(self, t: float) -> float | None:
        window = self._maintenance_window(t)
        return None if window is None else window[1]

    def next_maintenance_start(self, t: float) -> float:
        if self.scenario.part != 3:
            return math.inf
        period = 24.0 * self.scenario.maintenance_every_days
        if t < period:
            return period
        n = math.floor(t / period)
        current = n * period
        if abs(t - current) < EPS:
            return current
        return (n + 1) * period

    def next_shift_start(self, t: float) -> float:
        if self.scenario.part == 2:
            day = math.floor(t / 24.0)
            start = day * 24.0
            end = start + self.scenario.shift_hours
            if start <= t < end:
                return t
            if t < start:
                return start
            return (day + 1) * 24.0

        w1s, w1e, w2s, w2e, cycle = self._part3_windows()
        n = math.floor(t / cycle)
        base = n * cycle
        offset = t - base

        if w1s <= offset < w1e or w2s <= offset < w2e:
            return t
        if offset < w1s:
            return base + w1s
        if offset < w2s:
            return base + w2s
        return base + cycle

    def _shift_state(self, t: float) -> tuple[bool, float, float]:
        if self.scenario.part == 2:
            day = math.floor(t / 24.0)
            start = day * 24.0
            end = start + self.scenario.shift_hours
            return start <= t < end, start, end

        w1s, w1e, w2s, w2e, cycle = self._part3_windows()
        n = math.floor(t / cycle)
        base = n * cycle
        offset = t - base

        if w1s <= offset < w1e:
            return True, base + w1s, base + w1e
        if w2s <= offset < w2e:
            return True, base + w2s, base + w2e
        if offset < w1s:
            return False, base + w1s, base + w1e
        if offset < w2s:
            return False, base + w2s, base + w2e
        return False, base + cycle, base + cycle + self.scenario.shift_hours

    def _maintenance_window(self, t: float) -> tuple[float, float] | None:
        if self.scenario.part != 3:
            return None
        period = 24.0 * self.scenario.maintenance_every_days
        if t < period:
            return None
        n = math.floor(t / period)
        start = n * period
        end = start + self.scenario.maintenance_h
        if n >= 1 and start <= t < end:
            return start, end
        return None

    def _part3_windows(self) -> tuple[float, float, float, float, float]:
        k = self.scenario.shift_hours
        g = self.scenario.handover_h
        off = max(0.0, 24.0 - (2.0 * k + g))
        cycle = 2.0 * k + g + off
        return 0.0, k, k + g, 2.0 * k + g, cycle


# ---------------------------------------------------------------------------
# Simulation controller
# ---------------------------------------------------------------------------


class SimulationController:
    def __init__(self, config: SimulationConfig, scenario: Scenario, seed: int = 1) -> None:
        self.config = config
        self.scenario = scenario
        self._validate_scenario()
        self.rng = random.Random(seed)
        self.scheduler = Scheduler()
        self.calendar = WorkCalendar(scenario)
        self.failure_model = FailureModel()
        self.queue_seq = 0
        self.last_event_time = 0.0

        self.robots = [self._create_robot(robot_id=i) for i in range(scenario.n_robots)]
        self.waiting_for_bench: Deque[int] = deque(range(scenario.n_robots))
        self.benches = [Bench(resource_id=i) for i in range(config.benches)]
        self.stage_systems = {name: StageSystem(stage_cfg) for name, stage_cfg in config.stages.items()}

        self.completed_count = 0
        self.total_attempts = 0
        self.false_positive_events = 0
        self.false_negative_events = 0

        if scenario.part == 3:
            period = 24.0 * scenario.maintenance_every_days
            self.next_maintenance_reset = period + scenario.maintenance_h
            self.maintenance_period = period
        else:
            self.next_maintenance_reset = math.inf
            self.maintenance_period = math.inf

    def run(self) -> Dict[str, float]:
        while self.completed_count < self.scenario.n_robots:
            self._dispatch_until_stable()

            if self.completed_count >= self.scenario.n_robots:
                break

            if self.scheduler.has_events():
                next_time = self.scheduler.advance_to_next_event()
                if next_time + EPS < self.last_event_time:
                    raise RuntimeError("event time moved backwards")
                self._apply_maintenance_resets_up_to(next_time)
                self.scheduler.time = next_time
                self.last_event_time = next_time

                for event in self.scheduler.pop_current_events():
                    if event.kind == EventKind.TASK_END:
                        self._handle_task_end(event)
                    elif event.kind == EventKind.REPAIR_DONE:
                        self._handle_repair_done(event)
                    else:
                        raise RuntimeError(f"unsupported event kind: {event.kind}")

                self._assert_invariants()
                continue

            next_work_time = self.calendar.next_work_start(self.scheduler.time + EPS)
            if next_work_time + EPS < self.scheduler.time:
                raise RuntimeError("calendar moved time backwards")
            self._apply_maintenance_resets_up_to(next_work_time)
            self.scheduler.time = next_work_time
            self.last_event_time = next_work_time
            self._assert_invariants()

        self._assert_invariants(final=True)
        return self._build_summary()

    def _validate_scenario(self) -> None:
        longest_task = max(
            self.config.load_h,
            self.config.unload_h,
            self.config.predebug_h,
            *(cfg.calibration_h + cfg.test_h for cfg in self.config.stages.values()),
        )
        if self.scenario.shift_hours + EPS < longest_task:
            raise ValueError(
                "shift_hours is shorter than the longest restartable task; the model would never complete"
            )

    def _create_robot(self, robot_id: int) -> Robot:
        defects = {
            "A": self.rng.random() < self.config.stages["A"].defect_rate,
            "B": self.rng.random() < self.config.stages["B"].defect_rate,
            "C": self.rng.random() < self.config.stages["C"].defect_rate,
            "D": self.rng.random() < self.config.latent_defect_rate_d,
        }
        hidden = {"D"} if defects["D"] else set()
        return Robot(robot_id=robot_id, defects=defects, hidden_defects=hidden)

    def _dispatch_until_stable(self) -> None:
        iterations = 0
        while True:
            iterations += 1
            if iterations > 100_000:
                raise RuntimeError("dispatch loop did not stabilize")

            changed = False
            changed |= self._assign_benches()
            changed |= self._dispatch_bench_operations()
            for stage in STAGES:
                changed |= self._dispatch_stage_queue(self.stage_systems[stage])
            if not changed:
                return

    def _assign_benches(self) -> bool:
        if not self.calendar.is_work_time(self.scheduler.time):
            return False
        changed = False
        for bench in self.benches:
            if not self.waiting_for_bench:
                break
            if not bench.can_accept_robot(self.scheduler.time):
                continue
            robot_id = self.waiting_for_bench.popleft()
            robot = self.robots[robot_id]
            bench.occupy(robot_id=robot_id, now=self.scheduler.time)
            robot.bench_id = bench.resource_id
            changed = True
        return changed

    def _dispatch_bench_operations(self) -> bool:
        if not self.calendar.is_work_time(self.scheduler.time):
            return False

        changed = False
        for bench in self.benches:
            robot_id = bench.occupied_robot_id
            if robot_id is None:
                continue

            robot = self.robots[robot_id]
            if robot.done or robot.active_operation is not None:
                continue

            operation = self._choose_operation(robot)
            if operation is None:
                continue

            if operation.kind in (OperationKind.LOAD, OperationKind.UNLOAD):
                changed |= self._start_bench_task(robot=robot, bench=bench, operation=operation)
                continue

            if robot.queued_stage is not None:
                continue

            queue_stage = operation.queue_stage
            if queue_stage is None:
                continue

            system = self.stage_systems[queue_stage]
            if not system.queue and system.has_available_capacity(self.scheduler.time):
                changed |= self._start_stage_task(robot=robot, operation=operation, stage_system=system)
            else:
                changed |= self._enqueue_stage_request(robot=robot, operation=operation, stage_system=system)

        return changed

    def _dispatch_stage_queue(self, stage_system: StageSystem) -> bool:
        if not self.calendar.is_work_time(self.scheduler.time):
            return False

        changed = False
        while stage_system.queue and stage_system.has_available_capacity(self.scheduler.time):
            request = stage_system.pop_request()
            if request is None:
                break

            robot = self.robots[request.robot_id]
            robot.queued_stage = None

            if robot.done or robot.active_operation is not None or robot.bench_id is None:
                continue

            current_operation = self._choose_operation(robot)
            if current_operation != request.operation:
                continue

            changed |= self._start_stage_task(robot=robot, operation=current_operation, stage_system=stage_system)

        return changed

    def _enqueue_stage_request(self, *, robot: Robot, operation: Operation, stage_system: StageSystem) -> bool:
        if robot.queued_stage is not None:
            return False

        priority_value = self._queue_priority(robot)
        if stage_system.config.queue_policy == QueuePolicy.FIFO:
            primary = 0
        else:
            primary = priority_value

        request = QueueRequest(
            primary_key=primary,
            secondary_key=self.queue_seq,
            robot_id=robot.robot_id,
            operation=operation,
        )
        self.queue_seq += 1

        if not stage_system.enqueue(request):
            return False

        robot.queued_stage = stage_system.name
        return True

    def _start_bench_task(self, *, robot: Robot, bench: Bench, operation: Operation) -> bool:
        if not self.calendar.is_work_time(self.scheduler.time):
            return False
        if bench.busy or bench.occupied_robot_id != robot.robot_id:
            return False

        window_end = self.calendar.current_work_window_end(self.scheduler.time)
        if window_end is None:
            return False

        full_h = self._operation_duration(robot, operation)
        available_h = window_end - self.scheduler.time
        if available_h <= EPS:
            return False

        run_h = min(full_h, available_h)
        if run_h < -EPS:
            raise RuntimeError("negative bench duration")
        outcome = TaskOutcome.COMPLETE if full_h <= available_h + EPS else TaskOutcome.INTERRUPTED

        bench.assign(robot_id=robot.robot_id, operation=operation)
        robot.active_operation = operation
        self.scheduler.schedule(
            time=self.scheduler.time + run_h,
            priority=20,
            kind=EventKind.TASK_END,
            robot_id=robot.robot_id,
            operation=operation,
            outcome=outcome,
            run_h=run_h,
            bench_id=bench.resource_id,
        )
        return True

    def _start_stage_task(self, *, robot: Robot, operation: Operation, stage_system: StageSystem) -> bool:
        if not self.calendar.is_work_time(self.scheduler.time):
            return False
        if robot.active_operation is not None:
            return False

        pair = stage_system.select_pair(self.scheduler.time)
        if pair is None:
            return False

        window_end = self.calendar.current_work_window_end(self.scheduler.time)
        if window_end is None:
            return False

        station, worker = pair
        full_h = self._operation_duration(robot, operation)
        available_h = window_end - self.scheduler.time
        if available_h <= EPS:
            return False

        failure_h = self.failure_model.sample_failure_time(
            station=station,
            duration_h=min(full_h, available_h),
            rng=self.rng,
        )

        run_h = min(full_h, available_h)
        outcome = TaskOutcome.COMPLETE if full_h <= available_h + EPS else TaskOutcome.INTERRUPTED
        if failure_h is not None and failure_h < run_h - EPS:
            run_h = max(failure_h, EPS)
            outcome = TaskOutcome.EQUIPMENT_FAILURE

        if run_h < -EPS:
            raise RuntimeError("negative stage duration")

        station.assign(robot_id=robot.robot_id, operation=operation)
        worker.assign(robot_id=robot.robot_id, operation=operation)
        robot.active_operation = operation

        self.scheduler.schedule(
            time=self.scheduler.time + run_h,
            priority=10,
            kind=EventKind.TASK_END,
            robot_id=robot.robot_id,
            operation=operation,
            outcome=outcome,
            run_h=run_h,
            stage=stage_system.name,
            worker_id=worker.resource_id,
            station_id=station.resource_id,
        )
        return True

    def _handle_task_end(self, event: Event) -> None:
        if event.robot_id is None or event.operation is None or event.outcome is None:
            raise RuntimeError("task event is missing required fields")

        robot = self.robots[event.robot_id]
        operation = event.operation

        if event.bench_id is not None:
            bench = self.benches[event.bench_id]
            if bench.current_robot_id != robot.robot_id:
                raise RuntimeError("bench release does not match robot")
            bench.release(event.run_h)

        station: Station | None = None
        if event.stage is not None:
            stage_system = self.stage_systems[event.stage]
            if event.station_id is None or event.worker_id is None:
                raise RuntimeError("stage event is missing resource ids")
            station = stage_system.stations[event.station_id]
            worker = stage_system.workers[event.worker_id]
            if station.current_robot_id != robot.robot_id or worker.current_robot_id != robot.robot_id:
                raise RuntimeError("resource release does not match robot")
            station.release(event.run_h)
            worker.release(event.run_h)

        robot.active_operation = None

        if event.outcome == TaskOutcome.EQUIPMENT_FAILURE:
            if station is None:
                raise RuntimeError("equipment failure without station")
            station.failures += 1
            station.down_until = self.scheduler.time + self.config.repair_h
            self.scheduler.schedule(
                time=station.down_until,
                priority=0,
                kind=EventKind.REPAIR_DONE,
                stage=station.stage,
                station_id=station.resource_id,
            )
            return

        if event.outcome == TaskOutcome.INTERRUPTED:
            return

        if operation.kind == OperationKind.LOAD:
            robot.loaded = True
            return

        if operation.kind == OperationKind.UNLOAD:
            robot.done = True
            robot.finish_time = self.scheduler.time
            self.completed_count += 1
            if robot.bench_id is None:
                raise RuntimeError("unloading robot without bench")
            bench = self.benches[robot.bench_id]
            bench.release_robot(self.scheduler.time)
            robot.bench_id = None
            return

        if operation.kind == OperationKind.PREDEBUG:
            robot.predebug_done = True
            return

        if operation.kind == OperationKind.TEST and operation.stage is not None:
            self._evaluate_test_result(robot=robot, stage=operation.stage)
            return

        raise RuntimeError(f"unexpected operation completion: {operation.label}")

    def _handle_repair_done(self, event: Event) -> None:
        if event.stage is None or event.station_id is None:
            raise RuntimeError("repair event is missing station identity")
        station = self.stage_systems[event.stage].stations[event.station_id]
        if station.down_until <= self.scheduler.time + EPS:
            station.down_until = 0.0

    def _evaluate_test_result(self, *, robot: Robot, stage: str) -> None:
        self.total_attempts += 1
        cfg = self.config.stages[stage]

        if stage in ABC_STAGES:
            has_problem = robot.defects[stage]
        else:
            has_problem = robot.has_hidden_defect()

        if has_problem:
            pass_probability = self.config.beta * cfg.operator_error_rate
        else:
            pass_probability = 1.0 - self.config.alpha * cfg.operator_error_rate

        passed = self.rng.random() < pass_probability
        robot.attempts[stage] += 1

        if has_problem and passed:
            self.false_negative_events += 1
            if stage in ABC_STAGES and robot.defects[stage]:
                robot.hidden_defects.add(stage)

        if (not has_problem) and (not passed):
            self.false_positive_events += 1

        if passed:
            robot.stage_pass[stage] = True
            return

        if robot.attempts[stage] >= self.config.max_attempts_per_stage:
            robot.eliminated = True

    def _choose_operation(self, robot: Robot) -> Operation | None:
        if robot.done:
            return None
        if not robot.loaded:
            return Operation(OperationKind.LOAD)
        if robot.eliminated:
            return Operation(OperationKind.UNLOAD)

        next_stage = robot.next_stage()
        if next_stage is None:
            return Operation(OperationKind.UNLOAD)

        if self._can_predebug(robot):
            if next_stage == "E":
                return Operation(OperationKind.PREDEBUG, stage="E")
            if next_stage == "C":
                c_system = self.stage_systems["C"]
                e_system = self.stage_systems["E"]
                if (not c_system.has_available_capacity(self.scheduler.time)) and e_system.has_available_capacity(self.scheduler.time):
                    return Operation(OperationKind.PREDEBUG, stage="E")

        return Operation(OperationKind.TEST, stage=next_stage)

    def _can_predebug(self, robot: Robot) -> bool:
        if not self.scenario.enable_predebug:
            return False
        if robot.predebug_done or robot.eliminated or robot.stage_pass["E"]:
            return False
        return robot.passed_count_abc() >= 2

    def _operation_duration(self, robot: Robot, operation: Operation) -> float:
        if operation.kind == OperationKind.LOAD:
            return self.config.load_h
        if operation.kind == OperationKind.UNLOAD:
            return self.config.unload_h
        if operation.kind == OperationKind.PREDEBUG:
            return self.config.predebug_h
        if operation.kind == OperationKind.TEST and operation.stage is not None:
            stage_cfg = self.config.stages[operation.stage]
            test_h = stage_cfg.test_h
            if operation.stage == "E" and robot.predebug_done:
                test_h *= 1.0 - self.config.predebug_reduce_e
            return stage_cfg.calibration_h + test_h
        raise RuntimeError(f"unsupported operation: {operation}")

    def _queue_priority(self, robot: Robot) -> int:
        remaining = sum(1 for stage in STAGES if not robot.stage_pass[stage])
        return remaining

    def _apply_maintenance_resets_up_to(self, t: float) -> None:
        if self.scenario.part != 3:
            return
        while self.next_maintenance_reset <= t + EPS:
            for system in self.stage_systems.values():
                system.reset_maintenance()
            self.next_maintenance_reset += self.maintenance_period

    def _assert_invariants(self, final: bool = False) -> None:
        if self.scheduler.time < -EPS:
            raise RuntimeError("negative simulation time")

        for robot in self.robots:
            if robot.done and robot.finish_time is None:
                raise RuntimeError("finished robot missing finish time")
            if robot.finish_time is not None and robot.finish_time > self.scheduler.time + EPS and not final:
                raise RuntimeError("robot finish time is in the future")
            if robot.active_operation is None and robot.queued_stage is not None and robot.done:
                raise RuntimeError("finished robot left in queue state")

        for bench in self.benches:
            if bench.productive_hours < -EPS or bench.occupied_calendar_hours < -EPS:
                raise RuntimeError("negative bench accounting")
            if bench.busy and bench.current_robot_id != bench.occupied_robot_id:
                raise RuntimeError("bench task robot does not match occupied robot")

        for system in self.stage_systems.values():
            for station in system.stations:
                if station.productive_hours < -EPS or station.cumulative_hours < -EPS:
                    raise RuntimeError("negative station accounting")
            for worker in system.workers:
                if worker.productive_hours < -EPS:
                    raise RuntimeError("negative worker accounting")

    def _build_summary(self) -> Dict[str, float]:
        makespan_h = max((robot.finish_time or 0.0) for robot in self.robots)
        makespan_days = makespan_h / 24.0
        work_hours = self.calendar.total_work_hours(makespan_h)

        qualified = [robot for robot in self.robots if robot.done and robot.is_qualified()]
        qualified_hidden = [robot for robot in qualified if robot.has_hidden_defect()]
        rejected = [robot for robot in self.robots if robot.eliminated]

        bench_productive = sum(bench.productive_hours for bench in self.benches)
        bench_occupancy = sum(bench.occupied_calendar_hours for bench in self.benches)

        summary: Dict[str, float] = {
            "makespan_h": makespan_h,
            "makespan_days": makespan_days,
            "makespan_days_ceiled": float(math.ceil(makespan_days)),
            "completed_count": float(self.completed_count),
            "qualified_count": float(len(qualified)),
            "rejected_count": float(len(rejected)),
            "hidden_qualified_count": float(len(qualified_hidden)),
            "omission_probability": len(qualified_hidden) / max(1.0, float(len(qualified))),
            "misjudgment_probability": self.false_positive_events / max(1.0, float(self.total_attempts)),
            "total_attempts": float(self.total_attempts),
            "false_positive_events": float(self.false_positive_events),
            "false_negative_events": float(self.false_negative_events),
            "bench_effective_work_ratio": bench_productive / max(EPS, float(len(self.benches)) * work_hours),
            "bench_occupancy_ratio": bench_occupancy / max(EPS, float(len(self.benches)) * makespan_h),
        }

        for stage, system in self.stage_systems.items():
            summary[f"stage_{stage}_station_utilization"] = system.effective_hours() / max(
                EPS, float(len(system.stations)) * work_hours
            )
            summary[f"stage_{stage}_worker_utilization"] = system.worker_hours() / max(
                EPS, float(len(system.workers)) * work_hours
            )
            summary[f"stage_{stage}_failures"] = float(system.failures())

        return summary


# ---------------------------------------------------------------------------
# Robot testing workflow helpers
# ---------------------------------------------------------------------------


def build_default_config() -> SimulationConfig:
    bands_a = (
        FailureBand(80.0, 0.0012),
        FailureBand(160.0, 0.0020),
        FailureBand(math.inf, 0.0035),
    )
    bands_b = (
        FailureBand(80.0, 0.0010),
        FailureBand(160.0, 0.0018),
        FailureBand(math.inf, 0.0030),
    )
    bands_c = (
        FailureBand(80.0, 0.0011),
        FailureBand(160.0, 0.0019),
        FailureBand(math.inf, 0.0032),
    )
    bands_e = (
        FailureBand(80.0, 0.0014),
        FailureBand(160.0, 0.0024),
        FailureBand(math.inf, 0.0040),
    )

    stage_configs = {
        "A": StageConfig(
            name="A",
            calibration_h=20.0 / 60.0,
            test_h=2.0,
            defect_rate=0.03,
            operator_error_rate=0.025,
            worker_count=1,
            station_count=1,
            queue_capacity=2,
            queue_policy=QueuePolicy.FIFO,
            failure_bands=bands_a,
        ),
        "B": StageConfig(
            name="B",
            calibration_h=15.0 / 60.0,
            test_h=1.5,
            defect_rate=0.025,
            operator_error_rate=0.03,
            worker_count=1,
            station_count=1,
            queue_capacity=2,
            queue_policy=QueuePolicy.FIFO,
            failure_bands=bands_b,
        ),
        "C": StageConfig(
            name="C",
            calibration_h=15.0 / 60.0,
            test_h=2.0,
            defect_rate=0.02,
            operator_error_rate=0.015,
            worker_count=1,
            station_count=1,
            queue_capacity=2,
            queue_policy=QueuePolicy.FIFO,
            failure_bands=bands_c,
        ),
        "E": StageConfig(
            name="E",
            calibration_h=30.0 / 60.0,
            test_h=2.5,
            defect_rate=0.0,
            operator_error_rate=0.02,
            worker_count=2,
            station_count=2,
            queue_capacity=4,
            queue_policy=QueuePolicy.PRIORITY,
            failure_bands=bands_e,
        ),
    }

    return SimulationConfig(stages=stage_configs, benches=2, repair_h=1.0)


def run_replications(
    config: SimulationConfig,
    scenario: Scenario,
    *,
    replications: int,
    seed0: int = 10_000,
) -> Dict[str, float]:
    accum: Dict[str, float] = {}
    for rep in range(replications):
        controller = SimulationController(config=config, scenario=scenario, seed=seed0 + rep * 9973)
        result = controller.run()
        for key, value in result.items():
            accum[key] = accum.get(key, 0.0) + value
    return {key: value / replications for key, value in accum.items()}


def optimize_shift_length(
    config: SimulationConfig,
    *,
    n_robots: int = 240,
    replications: int = 20,
    k_min: float = 9.0,
    k_max: float = 12.0,
    k_step: float = 0.5,
    seed0: int = 30_000,
) -> tuple[float, Dict[str, float], Dict[float, Dict[str, float]]]:
    all_results: Dict[float, Dict[str, float]] = {}
    k = k_min
    idx = 0
    while k <= k_max + EPS:
        shift_h = round(k, 10)
        scenario = Scenario(
            n_robots=n_robots,
            part=3,
            shift_hours=shift_h,
            handover_h=0.5,
            maintenance_every_days=7,
            maintenance_h=4.0,
            enable_predebug=True,
        )
        all_results[shift_h] = run_replications(
            config=config,
            scenario=scenario,
            replications=replications,
            seed0=seed0 + idx * 100_000,
        )
        idx += 1
        k += k_step

    best_k = min(
        all_results,
        key=lambda key: (
            all_results[key]["makespan_days"],
            all_results[key]["omission_probability"],
        ),
    )
    return best_k, all_results[best_k], all_results


# ---------------------------------------------------------------------------
# Main simulation runner
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized hybrid robot testing DES")
    parser.add_argument("--robots", type=int, default=120, help="Number of robots to simulate")
    parser.add_argument("--part", type=int, choices=(2, 3), default=2, help="Scenario type")
    parser.add_argument("--shift-hours", type=float, default=12.0, help="Shift length in hours")
    parser.add_argument("--reps", type=int, default=1, help="Replication count")
    parser.add_argument("--seed", type=int, default=20260328, help="Base random seed")
    parser.add_argument("--enable-predebug", action="store_true", help="Enable E pre-debug flow")
    parser.add_argument("--optimize-k", action="store_true", help="Run part-3 K sweep instead of a single scenario")
    parser.add_argument("--k-min", type=float, default=9.0, help="Minimum K for optimization")
    parser.add_argument("--k-max", type=float, default=12.0, help="Maximum K for optimization")
    parser.add_argument("--k-step", type=float, default=0.5, help="Step size for optimization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_default_config()

    if args.optimize_k:
        best_k, best_result, all_results = optimize_shift_length(
            config,
            n_robots=args.robots,
            replications=args.reps,
            k_min=args.k_min,
            k_max=args.k_max,
            k_step=args.k_step,
            seed0=args.seed,
        )
        payload = {
            "best_k": best_k,
            "best_result": best_result,
            "all_results": {str(key): value for key, value in all_results.items()},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    scenario = Scenario(
        n_robots=args.robots,
        part=args.part,
        shift_hours=args.shift_hours,
        handover_h=0.5,
        maintenance_every_days=7,
        maintenance_h=4.0,
        enable_predebug=args.enable_predebug,
    )

    if args.reps > 1:
        result = run_replications(config=config, scenario=scenario, replications=args.reps, seed0=args.seed)
    else:
        result = SimulationController(config=config, scenario=scenario, seed=args.seed).run()

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
