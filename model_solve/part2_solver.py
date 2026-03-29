#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


EPS = 1e-9
STAGES_ABC = ("A", "B", "C")
STAGES_ALL = ("A", "B", "C", "E")


@dataclass
class FailureBand:
    max_hour: float
    rate_per_hour: float


@dataclass
class SimulationParams:
    p: Dict[str, float] = field(
        default_factory=lambda: {"A": 0.03, "B": 0.025, "C": 0.02, "D": 0.002}
    )
    q: Dict[str, float] = field(
        default_factory=lambda: {"A": 0.025, "B": 0.03, "C": 0.015, "E": 0.02}
    )
    alpha: float = 0.6
    beta: float = 0.4
    delta: Dict[str, float] = field(
        default_factory=lambda: {"A": 20 / 60, "B": 15 / 60, "C": 15 / 60, "E": 30 / 60}
    )
    theta: Dict[str, float] = field(
        default_factory=lambda: {"A": 2.0, "B": 1.5, "C": 2.0, "E": 2.5}
    )
    load_time: float = 0.5
    unload_time: float = 0.5
    benches: int = 2
    stations: Tuple[str, ...] = ("A", "B", "C", "E")
    failure_rate_table: Dict[str, List[FailureBand]] = field(
        default_factory=lambda: {
            "A": [FailureBand(80, 0.0012), FailureBand(160, 0.0020), FailureBand(math.inf, 0.0035)],
            "B": [FailureBand(80, 0.0010), FailureBand(160, 0.0018), FailureBand(math.inf, 0.0030)],
            "C": [FailureBand(80, 0.0011), FailureBand(160, 0.0019), FailureBand(math.inf, 0.0032)],
            "E": [FailureBand(80, 0.0014), FailureBand(160, 0.0024), FailureBand(math.inf, 0.0040)],
        }
    )
    repair_time_after_failure: float = 1.0


@dataclass
class Scenario:
    n_robots: int
    shift_hours: float = 12.0


@dataclass
class Robot:
    rid: int
    bench_id: Optional[int] = None
    loaded: bool = False
    eliminated: bool = False
    done: bool = False
    finish_time: Optional[float] = None
    stage_pass: Dict[str, bool] = field(default_factory=lambda: {"A": False, "B": False, "C": False, "E": False})
    attempts: Dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0, "E": 0})
    defects: Dict[str, bool] = field(default_factory=dict)
    hidden_defects: Dict[str, bool] = field(default_factory=lambda: {"A": False, "B": False, "C": False, "D": False})
    active_task: Optional[str] = None
    omission_events: int = 0
    misjudge_events: int = 0


@dataclass
class Bench:
    bid: int
    robot_id: Optional[int] = None
    occupied_since: Optional[float] = None
    occupied_calendar_hours: float = 0.0
    productive_hours: float = 0.0


@dataclass
class Station:
    name: str
    status: str = "idle"
    busy_robot: Optional[int] = None
    cumulative_hours: float = 0.0
    productive_hours: float = 0.0
    failures: int = 0
    down_until: float = 0.0


@dataclass(order=True)
class Event:
    time: float
    seq: int
    kind: str = field(compare=False)
    rid: Optional[int] = field(compare=False, default=None)
    op: Optional[str] = field(compare=False, default=None)
    outcome: Optional[str] = field(compare=False, default=None)
    run_time: float = field(compare=False, default=0.0)


class WorkCalendar:
    def __init__(self, scenario: Scenario):
        self.s = scenario

    def is_work_time(self, t: float) -> bool:
        day = math.floor(t / 24.0)
        start = day * 24.0
        end = start + self.s.shift_hours
        return start <= t < end

    def current_work_window_end(self, t: float) -> Optional[float]:
        if not self.is_work_time(t):
            return None
        day = math.floor(t / 24.0)
        return day * 24.0 + self.s.shift_hours

    def next_shift_start(self, t: float) -> float:
        day = math.floor(t / 24.0)
        start = day * 24.0
        end = start + self.s.shift_hours
        if start <= t < end:
            return t
        if t < start:
            return start
        return (day + 1) * 24.0

    def next_work_start(self, t: float) -> float:
        probe = t
        for _ in range(10000):
            if self.is_work_time(probe):
                return probe
            probe = self.next_shift_start(probe + EPS)
        raise RuntimeError("next_work_start did not converge")

    def total_work_hours(self, t_end: float) -> float:
        t = 0.0
        total = 0.0
        while t < t_end - EPS:
            t = self.next_work_start(t)
            if t >= t_end - EPS:
                break
            window_end = self.current_work_window_end(t)
            if window_end is None:
                t += EPS
                continue
            total += max(0.0, min(window_end, t_end) - t)
            t = window_end + EPS
        return total


class FactorySimulationPart2:
    def __init__(self, params: SimulationParams, scenario: Scenario, seed: int = 1234):
        self.p = params
        self.sc = scenario
        self.rng = random.Random(seed)
        self.calendar = WorkCalendar(scenario)

        self.time = 0.0
        self.seq = 0
        self.events: List[Event] = []

        self.robots: List[Robot] = [self._init_robot(i) for i in range(self.sc.n_robots)]
        self.waiting_for_bench: Deque[int] = deque(range(self.sc.n_robots))

        self.benches: List[Bench] = [Bench(i) for i in range(self.p.benches)]
        self.stations: Dict[str, Station] = {name: Station(name) for name in self.p.stations}

        self.total_test_attempts = 0
        self.total_omission_events = 0
        self.total_misjudge_events = 0

    def _init_robot(self, rid: int) -> Robot:
        robot = Robot(rid=rid)
        robot.defects = {
            "A": self.rng.random() < self.p.p["A"],
            "B": self.rng.random() < self.p.p["B"],
            "C": self.rng.random() < self.p.p["C"],
            "D": self.rng.random() < self.p.p["D"],
        }
        robot.hidden_defects["D"] = robot.defects["D"]
        return robot

    def _push_event(self, event: Event) -> None:
        heapq.heappush(self.events, event)

    def _pop_events_at_current_time(self) -> List[Event]:
        out = []
        while self.events and abs(self.events[0].time - self.time) < EPS:
            out.append(heapq.heappop(self.events))
        return out

    def _failure_rate(self, station_name: str, cumulative_hours: float) -> float:
        bands = self.p.failure_rate_table[station_name]
        for band in bands:
            if cumulative_hours < band.max_hour:
                return band.rate_per_hour
        return bands[-1].rate_per_hour

    def _assign_benches_if_possible(self) -> bool:
        changed = False
        if not self.calendar.is_work_time(self.time):
            return False

        for bench in self.benches:
            if not self.waiting_for_bench:
                break
            if bench.robot_id is not None:
                continue
            rid = self.waiting_for_bench.popleft()
            robot = self.robots[rid]
            bench.robot_id = rid
            bench.occupied_since = self.time
            robot.bench_id = bench.bid
            changed = True
            self._start_task(robot, "LOAD")

        return changed

    def _robot_required_op(self, robot: Robot) -> Optional[str]:
        if robot.done:
            return None
        if not robot.loaded:
            return "LOAD"
        if robot.eliminated:
            return "UNLOAD"
        for stage in STAGES_ALL:
            if not robot.stage_pass[stage]:
                return f"TEST_{stage}"
        return "UNLOAD"

    def _operation_duration(self, robot: Robot, op: str) -> float:
        if op == "LOAD":
            return self.p.load_time
        if op == "UNLOAD":
            return self.p.unload_time
        stage = op.split("_")[1]
        return self.p.delta[stage] + self.p.theta[stage]

    def _start_task(self, robot: Robot, op: str) -> bool:
        if robot.active_task is not None or not self.calendar.is_work_time(self.time):
            return False

        window_end = self.calendar.current_work_window_end(self.time)
        if window_end is None:
            return False

        max_run = window_end - self.time
        if max_run <= EPS:
            return False

        duration = self._operation_duration(robot, op)
        run_time = min(duration, max_run)
        outcome = "complete" if duration <= max_run + EPS else "interrupt_shift"

        station_name: Optional[str] = None
        if op.startswith("TEST_"):
            station_name = op.split("_")[1]

        if station_name is not None:
            station = self.stations[station_name]
            if station.status != "idle" or station.down_until > self.time + EPS:
                return False
            rate = self._failure_rate(station_name, station.cumulative_hours)
            if rate > 0:
                t_fail = self.rng.expovariate(rate)
                if t_fail < run_time - EPS:
                    run_time = max(t_fail, EPS)
                    outcome = "interrupt_failure"
            station.status = "busy"
            station.busy_robot = robot.rid

        robot.active_task = op
        self._push_event(
            Event(
                time=self.time + run_time,
                seq=self.seq,
                kind="TASK_END",
                rid=robot.rid,
                op=op,
                outcome=outcome,
                run_time=run_time,
            )
        )
        self.seq += 1
        return True

    def _handle_test_outcome(self, robot: Robot, stage: str) -> None:
        self.total_test_attempts += 1

        if stage in STAGES_ABC:
            has_problem = robot.defects[stage]
        else:
            has_problem = (
                robot.hidden_defects["A"]
                or robot.hidden_defects["B"]
                or robot.hidden_defects["C"]
                or robot.hidden_defects["D"]
            )

        q = self.p.q[stage]
        pass_prob = self.p.beta * q if has_problem else 1.0 - self.p.alpha * q
        passed = self.rng.random() < pass_prob

        if has_problem and passed:
            robot.omission_events += 1
            self.total_omission_events += 1
            if stage in STAGES_ABC:
                robot.hidden_defects[stage] = True

        if (not has_problem) and (not passed):
            robot.misjudge_events += 1
            self.total_misjudge_events += 1

        robot.attempts[stage] += 1
        if passed:
            robot.stage_pass[stage] = True
        elif robot.attempts[stage] >= 2:
            robot.eliminated = True

    def _release_bench(self, robot: Robot) -> None:
        if robot.bench_id is None:
            return
        bench = self.benches[robot.bench_id]
        if bench.occupied_since is not None:
            bench.occupied_calendar_hours += self.time - bench.occupied_since
        bench.robot_id = None
        bench.occupied_since = None
        robot.bench_id = None

    def _handle_task_end(self, event: Event) -> None:
        robot = self.robots[event.rid]  # type: ignore[index]
        op = event.op
        assert op is not None

        if op in ("LOAD", "UNLOAD"):
            assert robot.bench_id is not None
            self.benches[robot.bench_id].productive_hours += event.run_time

        station_name: Optional[str] = None
        if op.startswith("TEST_"):
            station_name = op.split("_")[1]

        if station_name is not None:
            station = self.stations[station_name]
            station.productive_hours += event.run_time
            station.cumulative_hours += event.run_time
            station.status = "idle"
            station.busy_robot = None

        robot.active_task = None

        if event.outcome == "interrupt_failure":
            assert station_name is not None
            station = self.stations[station_name]
            station.failures += 1
            station.status = "down"
            station.down_until = self.time + self.p.repair_time_after_failure
            self._push_event(
                Event(
                    time=station.down_until,
                    seq=self.seq,
                    kind="REPAIR_DONE",
                    op=station_name,
                )
            )
            self.seq += 1
            return

        if event.outcome == "interrupt_shift":
            return

        if op == "LOAD":
            robot.loaded = True
            return
        if op == "UNLOAD":
            robot.done = True
            robot.finish_time = self.time
            self._release_bench(robot)
            return

        stage = op.split("_")[1]
        self._handle_test_outcome(robot, stage)

    def _handle_repair_done(self, event: Event) -> None:
        station_name = event.op
        assert station_name is not None
        station = self.stations[station_name]
        if station.status == "down" and station.down_until <= self.time + EPS:
            station.status = "idle"

    def _dispatch(self) -> bool:
        if not self.calendar.is_work_time(self.time):
            return False

        changed_any = False
        while True:
            changed = False
            if self._assign_benches_if_possible():
                changed = True

            candidates = [robot for robot in self.robots if robot.bench_id is not None and not robot.done and robot.active_task is None]
            candidates.sort(key=lambda robot: (robot.bench_id if robot.bench_id is not None else 999, robot.rid))

            for robot in candidates:
                op = self._robot_required_op(robot)
                if op is None:
                    continue
                if self._start_task(robot, op):
                    changed = True

            if not changed:
                break
            changed_any = True

        return changed_any

    def _done_count(self) -> int:
        return sum(1 for robot in self.robots if robot.done)

    def run(self) -> Dict[str, float]:
        while self._done_count() < self.sc.n_robots:
            self._dispatch()

            if self._done_count() >= self.sc.n_robots:
                break

            if self.events:
                self.time = self.events[0].time
                for event in self._pop_events_at_current_time():
                    if event.kind == "TASK_END":
                        self._handle_task_end(event)
                    elif event.kind == "REPAIR_DONE":
                        self._handle_repair_done(event)
                    else:
                        raise RuntimeError(f"Unknown event kind: {event.kind}")
                continue

            self.time = self.calendar.next_work_start(self.time + EPS)

        makespan_h = max((robot.finish_time or 0.0) for robot in self.robots)
        makespan_days = makespan_h / 24.0
        makespan_days_ceiled = math.ceil(makespan_days)

        qualified = [
            robot
            for robot in self.robots
            if robot.done and (not robot.eliminated) and robot.stage_pass["A"] and robot.stage_pass["B"] and robot.stage_pass["C"] and robot.stage_pass["E"]
        ]
        qualified_count = len(qualified)
        qualified_with_hidden_problem = sum(
            1
            for robot in qualified
            if robot.hidden_defects["A"] or robot.hidden_defects["B"] or robot.hidden_defects["C"] or robot.hidden_defects["D"]
        )

        omission_probability = qualified_with_hidden_problem / max(1, qualified_count)
        misjudgment_probability = self.total_misjudge_events / max(1, self.total_test_attempts)
        work_hours = self.calendar.total_work_hours(makespan_h)

        station_util = {
            f"station_{name}": station.productive_hours / max(work_hours, EPS)
            for name, station in self.stations.items()
        }
        bench_effective = sum(bench.productive_hours for bench in self.benches) / max(len(self.benches) * work_hours, EPS)
        bench_occupancy = sum(bench.occupied_calendar_hours for bench in self.benches) / max(len(self.benches) * makespan_h, EPS)

        results = {
            "makespan_hours": makespan_h,
            "makespan_days": makespan_days,
            "makespan_days_ceiled": float(makespan_days_ceiled),
            "qualified_count": float(qualified_count),
            "omission_probability": omission_probability,
            "misjudgment_probability": misjudgment_probability,
            "total_omission_events": float(self.total_omission_events),
            "total_misjudge_events": float(self.total_misjudge_events),
            "total_test_attempts": float(self.total_test_attempts),
            "bench_effective_work_ratio": bench_effective,
            "bench_occupancy_ratio": bench_occupancy,
        }
        results.update(station_util)
        return results


def run_replications(params: SimulationParams, scenario: Scenario, replications: int, seed0: int) -> Dict[str, float]:
    sums: Dict[str, float] = defaultdict(float)
    for replication in range(replications):
        simulation = FactorySimulationPart2(params, scenario, seed=seed0 + replication * 9973)
        result = simulation.run()
        for key, value in result.items():
            sums[key] += value
    return {key: value / replications for key, value in sums.items()}


def print_batch_result(title: str, result: Dict[str, float]) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"makespan_hours                 = {result['makespan_hours']:.4f}")
    print(f"makespan_days                  = {result['makespan_days']:.4f}")
    print(f"makespan_days_ceiled           = {result['makespan_days_ceiled']:.4f}")
    print(f"qualified_count                = {result['qualified_count']:.4f}")
    print(f"omission_probability           = {result['omission_probability']:.8f}")
    print(f"misjudgment_probability        = {result['misjudgment_probability']:.8f}")
    print(f"total_omission_events          = {result['total_omission_events']:.4f}")
    print(f"total_misjudge_events          = {result['total_misjudge_events']:.4f}")
    print(f"total_test_attempts            = {result['total_test_attempts']:.4f}")
    print(f"bench_effective_work_ratio     = {result['bench_effective_work_ratio']:.6f}")
    print(f"bench_occupancy_ratio          = {result['bench_occupancy_ratio']:.6f}")
    print(f"station_A_effective_ratio      = {result['station_A']:.6f}")
    print(f"station_B_effective_ratio      = {result['station_B']:.6f}")
    print(f"station_C_effective_ratio      = {result['station_C']:.6f}")
    print(f"station_E_effective_ratio      = {result['station_E']:.6f}")


def run_part2_simulation(replications: int = 100, seed0: int = 1000) -> Dict[str, float]:
    params = SimulationParams()
    scenario = Scenario(n_robots=120, shift_hours=12.0)
    result = run_replications(params, scenario, replications=replications, seed0=seed0)
    print_batch_result(f"Part 2: 120 Robots (Average over {replications} replications)", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Part 2 solver extracted from simulation.py")
    parser.add_argument("--replications", type=int, default=100, help="Number of Monte Carlo replications.")
    parser.add_argument("--seed0", type=int, default=1000, help="Base random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_part2_simulation(replications=args.replications, seed0=args.seed0)
