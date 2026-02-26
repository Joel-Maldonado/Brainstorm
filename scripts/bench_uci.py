#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

_CHESS_IMPORT_ERROR: Exception | None = None
try:
    import chess
    import chess.engine
except Exception as exc:  # pragma: no cover - exercised in dependency checks
    chess = None  # type: ignore[assignment]
    _CHESS_IMPORT_ERROR = exc

POSITIONS: Dict[str, str] = {
    "startpos": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "middlegame": "r2q1rk1/pp1b1ppp/2n1pn2/2bp4/2P5/2NP1NP1/PP2PPBP/R1BQ1RK1 w - - 0 8",
    "tactical": "2r2rk1/pp1n1pp1/2p1pn1p/2Pp4/3P4/2N1PN2/PP3PPP/2RR2K1 w - - 0 15",
    "endgame": "8/5pk1/3p1np1/2pPp3/2P1P3/3N1P2/5K1P/8 w - - 0 40",
    "king_attack": "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PPQ2PPP/R1B2RK1 w - - 0 9",
    "imbalanced": "r2q1rk1/pbpn1pp1/1p1bpn1p/3p4/3P4/2NBPN2/PPQ2PPP/R1B2RK1 w - - 2 10",
}
THREAD_DEFAULT_CAP = 8
MODEL_ALIASES: Dict[str, str] = {
    "fast": "fast",
    "balanced": "balanced",
    "accurate": "accurate",
    "small": "fast",
    "hybrid_root": "balanced",
    "large": "accurate",
}


@dataclass(frozen=True)
class LimitSpec:
    kind: str
    value: int

    @property
    def label(self) -> str:
        if self.kind == "depth":
            return f"depth={self.value}"
        return f"movetime={self.value}ms"

    def to_limit(self) -> chess.engine.Limit:
        if self.kind == "depth":
            return chess.engine.Limit(depth=self.value)
        return chess.engine.Limit(time=self.value / 1000.0)


@dataclass(frozen=True)
class BenchConfig:
    model: str
    threads: int
    hash_mb: int
    device: str


@dataclass(frozen=True)
class BenchCase:
    config: BenchConfig
    limit: LimitSpec
    position_name: str
    fen: str


def require_python_chess() -> None:
    if chess is not None:
        return

    detail = f": {_CHESS_IMPORT_ERROR}" if _CHESS_IMPORT_ERROR else ""
    raise SystemExit(
        "python-chess is required for benchmark runs. Install with `pip install python-chess`"
        f"{detail}"
    )


def _prepare_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("LIBTORCH_USE_PYTORCH", "1")

    if not env.get("DYLD_LIBRARY_PATH"):
        try:
            import torch  # type: ignore

            torch_lib = pathlib.Path(torch.__file__).resolve().parent / "lib"
            env["DYLD_LIBRARY_PATH"] = str(torch_lib)
        except Exception:
            pass

    return env


def parse_csv_items(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return items


def parse_csv_ints(value: str) -> List[int]:
    parsed: List[int] = []
    for item in parse_csv_items(value):
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer: {item}") from exc
    return parsed


def default_threads() -> int:
    return max(1, min(os.cpu_count() or 1, THREAD_DEFAULT_CAP))


def parse_thread_values(value: str) -> List[int]:
    parsed: List[int] = []
    auto_value = default_threads()
    for item in parse_csv_items(value):
        if item.lower() == "auto":
            parsed.append(auto_value)
            continue
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid thread value: {item}") from exc
    return parsed


def normalize_model_name(value: str) -> Tuple[str, str | None]:
    raw = value.strip()
    token = raw.lower()
    canonical = MODEL_ALIASES.get(token)
    if canonical is None:
        raise argparse.ArgumentTypeError(f"invalid model value: {value}")
    if token != canonical:
        return canonical, f"model `{raw}` is deprecated, use `{canonical}`"
    return canonical, None


def dedupe_sorted(values: Iterable[int]) -> List[int]:
    return sorted(set(values))


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    rank = (len(values) - 1) * p
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return values[lo]

    frac = rank - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def aggregate_numeric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "stdev": 0.0,
        }

    sorted_values = sorted(values)
    return {
        "count": float(len(sorted_values)),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "mean": float(statistics.fmean(sorted_values)),
        "median": float(percentile(sorted_values, 0.5)),
        "p95": float(percentile(sorted_values, 0.95)),
        "stdev": float(statistics.pstdev(sorted_values)) if len(sorted_values) > 1 else 0.0,
    }


def aggregate_records(records: Sequence[dict]) -> dict:
    elapsed_ms = [float(row["elapsed_ms"]) for row in records]
    nodes = [float(row["nodes"]) for row in records]
    depth = [float(row["depth"]) for row in records]
    knps = [float(row["knps"]) for row in records]

    return {
        "samples": len(records),
        "elapsed_ms": aggregate_numeric(elapsed_ms),
        "nodes": aggregate_numeric(nodes),
        "depth": aggregate_numeric(depth),
        "knps": aggregate_numeric(knps),
    }


def configure_engine(engine: chess.engine.SimpleEngine, config: BenchConfig) -> None:
    options = {
        "Hash": int(config.hash_mb),
        "Threads": int(config.threads),
        "Model": config.model,
        "Device": config.device,
        "DebugLog": False,
    }
    engine.configure(options)


def run_single(engine: chess.engine.SimpleEngine, case: BenchCase) -> dict:
    board = chess.Board(case.fen)
    t0 = time.perf_counter()
    result = engine.play(board, case.limit.to_limit(), info=chess.engine.INFO_ALL)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    info = result.info
    nodes = int(info.get("nodes") or 0)
    depth = int(info.get("depth") or 0)
    nps = int(info.get("nps") or 0)
    elapsed_s = max(elapsed_ms / 1000.0, 1e-9)
    knps = (nps / 1000.0) if nps > 0 else (nodes / elapsed_s) / 1000.0

    return {
        "model": case.config.model,
        "threads": case.config.threads,
        "hash_mb": case.config.hash_mb,
        "device": case.config.device,
        "limit_kind": case.limit.kind,
        "limit_value": case.limit.value,
        "limit": case.limit.label,
        "position": case.position_name,
        "move": str(result.move),
        "elapsed_ms": round(elapsed_ms, 3),
        "depth": depth,
        "nodes": nodes,
        "nps": nps,
        "knps": round(knps, 3),
        "score": str(info.get("score")) if info.get("score") is not None else None,
    }


def format_config(config: BenchConfig) -> str:
    return (
        f"model={config.model} threads={config.threads} "
        f"hash={config.hash_mb} device={config.device}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive UCI speed benchmark suite for Brainstorm"
    )
    parser.add_argument("--engine", default="./brainstorm", help="Engine binary path")
    parser.add_argument(
        "--models",
        default="fast,balanced,accurate",
        help="Comma-separated model list (legacy aliases: small, hybrid_root, large)",
    )
    parser.add_argument(
        "--threads",
        default="auto",
        help="Comma-separated thread counts (`auto` resolves to min(cpu_count, 8))",
    )
    parser.add_argument(
        "--hash-mb",
        default="64",
        help="Comma-separated hash sizes in MB",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Evaluator device",
    )
    parser.add_argument(
        "--depths",
        default="4,6",
        help="Comma-separated depth limits",
    )
    parser.add_argument(
        "--movetimes-ms",
        default="100,250,500",
        help="Comma-separated movetime limits in milliseconds",
    )
    parser.add_argument(
        "--positions",
        default=",".join(POSITIONS.keys()),
        help="Comma-separated position keys",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Measured runs per case")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per case")
    parser.add_argument(
        "--output-dir",
        default="results/bench_speed",
        help="Directory root for benchmark artifacts",
    )
    parser.add_argument("--tag", default="", help="Optional label appended to run directory")

    # Backward-compat flags used by older automation.
    parser.add_argument("--depth", type=int, default=None, help="Single depth benchmark")
    parser.add_argument(
        "--movetime-ms",
        type=int,
        default=None,
        help="Single movetime benchmark (ms)",
    )

    args = parser.parse_args()
    require_python_chess()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    raw_models = parse_csv_items(args.models)
    models: List[str] = []
    seen_models = set()
    for raw_model in raw_models:
        model, warning = normalize_model_name(raw_model)
        if warning:
            print(f"[bench] warning: {warning}", file=sys.stderr)
        if model not in seen_models:
            models.append(model)
            seen_models.add(model)
    thread_values = parse_thread_values(args.threads)
    hash_values = parse_csv_ints(args.hash_mb)

    depth_values = parse_csv_ints(args.depths)
    if args.depth is not None:
        depth_values.append(args.depth)
    depth_values = dedupe_sorted(v for v in depth_values if v > 0)

    movetime_values = parse_csv_ints(args.movetimes_ms)
    if args.movetime_ms is not None:
        movetime_values.append(args.movetime_ms)
    movetime_values = dedupe_sorted(v for v in movetime_values if v > 0)

    if not depth_values and not movetime_values:
        raise SystemExit("at least one depth or movetime limit is required")

    requested_positions = parse_csv_items(args.positions)
    missing = [name for name in requested_positions if name not in POSITIONS]
    if missing:
        raise SystemExit(f"unknown positions: {', '.join(missing)}")

    limits = [LimitSpec("depth", depth) for depth in depth_values]
    limits.extend(LimitSpec("movetime", ms) for ms in movetime_values)

    configs: List[BenchConfig] = []
    for model in models:
        for threads in thread_values:
            if threads < 1:
                raise SystemExit(f"threads must be >= 1, got {threads}")
            for hash_mb in hash_values:
                if hash_mb < 1:
                    raise SystemExit(f"hash-mb must be >= 1, got {hash_mb}")
                configs.append(
                    BenchConfig(model=model, threads=threads, hash_mb=hash_mb, device=args.device)
                )

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag_suffix = f"-{args.tag}" if args.tag else ""
    run_dir = pathlib.Path(args.output_dir).resolve() / f"{timestamp}{tag_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    engine_path = str(pathlib.Path(args.engine).resolve())
    env = _prepare_env()

    all_records: List[dict] = []
    samples_path = run_dir / "samples.jsonl"

    total_cases = len(configs) * len(limits) * len(requested_positions)
    completed_cases = 0

    with samples_path.open("w", encoding="utf-8") as sample_out:
        for config in configs:
            with chess.engine.SimpleEngine.popen_uci(engine_path, env=env) as engine:
                configure_engine(engine, config)

                for limit in limits:
                    for position_name in requested_positions:
                        fen = POSITIONS[position_name]
                        case = BenchCase(
                            config=config,
                            limit=limit,
                            position_name=position_name,
                            fen=fen,
                        )

                        for _ in range(args.warmup):
                            _ = run_single(engine, case)

                        for repeat_index in range(args.repeats):
                            record = run_single(engine, case)
                            record["repeat"] = repeat_index
                            all_records.append(record)
                            sample_out.write(json.dumps(record) + "\n")

                        completed_cases += 1
                        print(
                            f"[{completed_cases:>4}/{total_cases}] "
                            f"{format_config(config)} {limit.label} {position_name}"
                        )

    grouped_by_case: Dict[Tuple[str, int, int, str, str, str], List[dict]] = {}
    grouped_by_config_limit: Dict[Tuple[str, int, int, str, str], List[dict]] = {}

    for row in all_records:
        case_key = (
            row["model"],
            int(row["threads"]),
            int(row["hash_mb"]),
            row["device"],
            row["limit"],
            row["position"],
        )
        grouped_by_case.setdefault(case_key, []).append(row)

        config_key = (
            row["model"],
            int(row["threads"]),
            int(row["hash_mb"]),
            row["device"],
            row["limit"],
        )
        grouped_by_config_limit.setdefault(config_key, []).append(row)

    case_summaries = []
    for key, rows in sorted(grouped_by_case.items()):
        model, threads, hash_mb, device, limit, position = key
        case_summaries.append(
            {
                "model": model,
                "threads": threads,
                "hash_mb": hash_mb,
                "device": device,
                "limit": limit,
                "position": position,
                "metrics": aggregate_records(rows),
            }
        )

    config_summaries = []
    for key, rows in sorted(grouped_by_config_limit.items()):
        model, threads, hash_mb, device, limit = key
        config_summaries.append(
            {
                "model": model,
                "threads": threads,
                "hash_mb": hash_mb,
                "device": device,
                "limit": limit,
                "metrics": aggregate_records(rows),
            }
        )

    summary = {
        "engine": engine_path,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "positions": requested_positions,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "samples_file": str(samples_path),
        "totals": {
            "samples": len(all_records),
            "cases": total_cases,
            "configs": len(configs),
            "limits": [limit.label for limit in limits],
        },
        "by_config_limit": config_summaries,
        "by_case": case_summaries,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTop configurations by median elapsed (lower is better):")
    ranked = sorted(
        config_summaries,
        key=lambda row: row["metrics"]["elapsed_ms"]["median"],
    )
    for row in ranked[: min(10, len(ranked))]:
        elapsed = row["metrics"]["elapsed_ms"]
        knps = row["metrics"]["knps"]
        depth = row["metrics"]["depth"]
        print(
            f"- model={row['model']} threads={row['threads']} hash={row['hash_mb']} "
            f"device={row['device']} {row['limit']}: "
            f"median={elapsed['median']:.2f}ms p95={elapsed['p95']:.2f}ms "
            f"knps_mean={knps['mean']:.2f} depth_mean={depth['mean']:.2f}"
        )

    print("\nArtifacts:")
    print(f"- samples: {samples_path}")
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
