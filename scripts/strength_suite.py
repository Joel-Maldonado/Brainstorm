#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Dict, List

import chess
import chess.engine

DEFAULT_SUITE: List[Dict[str, object]] = [
    {
        "id": "M1-W-1",
        "fen": "k7/8/1QK5/8/8/8/8/8 w - - 0 1",
        "best_moves": ["b6b7"],
    },
    {
        "id": "M1-W-2",
        "fen": "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
        "best_moves": ["f7g7"],
    },
    {
        "id": "M1-B-1",
        "fen": "8/8/8/8/8/1qk5/8/K7 b - - 0 1",
        "best_moves": ["b3b2"],
    },
    {
        "id": "M1-B-4",
        "fen": "8/8/8/8/8/1k6/2q5/K7 b - - 0 1",
        "best_moves": ["c2c1"],
    },
]


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


def _load_suite(path: pathlib.Path | None) -> List[Dict[str, object]]:
    if path is None:
        return DEFAULT_SUITE

    suite: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            suite.append(json.loads(line))
    return suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tactical strength regression suite")
    parser.add_argument("--engine", default="./brainstorm", help="Engine binary path")
    parser.add_argument("--time-ms", type=int, default=2000, help="Time per position in ms")
    parser.add_argument(
        "--suite-jsonl",
        type=pathlib.Path,
        default=None,
        help="Optional JSONL suite file with {id, fen, best_moves}",
    )
    args = parser.parse_args()

    suite = _load_suite(args.suite_jsonl)
    engine_path = str(pathlib.Path(args.engine).resolve())

    solved = 0
    total = len(suite)

    with chess.engine.SimpleEngine.popen_uci(engine_path, env=_prepare_env()) as engine:
        for item in suite:
            test_id = str(item["id"])
            fen = str(item["fen"])
            expected = {str(move) for move in item.get("best_moves", [])}

            board = chess.Board(fen)
            t0 = time.perf_counter()
            result = engine.play(
                board,
                chess.engine.Limit(time=args.time_ms / 1000.0),
                info=chess.engine.INFO_ALL,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            move = str(result.move)
            is_solved = not expected or move in expected
            solved += int(is_solved)

            payload = {
                "id": test_id,
                "fen": fen,
                "move": move,
                "expected": sorted(expected),
                "solved": is_solved,
                "elapsed_ms": round(elapsed_ms, 1),
                "depth": result.info.get("depth"),
                "nodes": result.info.get("nodes"),
                "score": str(result.info.get("score")),
            }
            print(json.dumps(payload))

    solve_rate = (solved / max(total, 1)) * 100.0
    print(json.dumps({"summary": {"solved": solved, "total": total, "solve_rate": round(solve_rate, 2)}}))


if __name__ == "__main__":
    main()
