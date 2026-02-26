#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from typing import Dict, Iterable, Tuple

import chess
import chess.engine

POSITIONS: Dict[str, str] = {
    "startpos": chess.STARTING_FEN,
    "middlegame": "r2q1rk1/pp1b1ppp/2n1pn2/2bp4/2P5/2NP1NP1/PP2PPBP/R1BQ1RK1 w - - 0 8",
    "tacticish": "2r2rk1/pp1n1pp1/2p1pn1p/2Pp4/3P4/2N1PN2/PP3PPP/2RR2K1 w - - 0 15",
}


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


def _run_limits(
    engine_path: str,
    limit: chess.engine.Limit,
    label: str,
    positions: Iterable[Tuple[str, str]],
) -> None:
    with chess.engine.SimpleEngine.popen_uci(engine_path, env=_prepare_env()) as engine:
        for name, fen in positions:
            board = chess.Board(fen)
            t0 = time.perf_counter()
            result = engine.play(board, limit, info=chess.engine.INFO_ALL)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            info = result.info
            nodes = int(info.get("nodes") or 0)
            depth = int(info.get("depth") or 0)
            knps = (nodes / max(elapsed_ms / 1000.0, 1e-9)) / 1000.0

            payload = {
                "label": label,
                "position": name,
                "move": str(result.move),
                "elapsed_ms": round(elapsed_ms, 1),
                "depth": depth,
                "nodes": nodes,
                "knps": round(knps, 2),
                "score": str(info.get("score")),
            }
            print(json.dumps(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Brainstorm UCI engine")
    parser.add_argument("--engine", default="./brainstorm", help="Engine binary path")
    parser.add_argument("--depth", type=int, default=5, help="Depth for depth benchmark")
    parser.add_argument(
        "--movetime-ms", type=int, default=2000, help="Movetime benchmark in milliseconds"
    )
    args = parser.parse_args()

    engine_path = str(pathlib.Path(args.engine).resolve())
    positions = list(POSITIONS.items())

    _run_limits(engine_path, chess.engine.Limit(depth=args.depth), f"depth={args.depth}", positions)
    _run_limits(
        engine_path,
        chess.engine.Limit(time=args.movetime_ms / 1000.0),
        f"movetime={args.movetime_ms}ms",
        positions,
    )


if __name__ == "__main__":
    main()
