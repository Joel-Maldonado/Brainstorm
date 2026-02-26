from __future__ import annotations

import os
import pathlib
import time
import unittest

import chess
import chess.engine


def _prepare_env() -> dict[str, str]:
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


class TimeControlTest(unittest.TestCase):
    def test_movetime_respects_budget(self) -> None:
        engine_path = pathlib.Path(__file__).resolve().parents[1] / "brainstorm"
        if not engine_path.exists():
            self.skipTest("engine binary not found")

        board = chess.Board("r2q1rk1/pp1b1ppp/2n1pn2/2bp4/2P5/2NP1NP1/PP2PPBP/R1BQ1RK1 w - - 0 8")
        movetimes_ms = [100, 500, 2000]

        with chess.engine.SimpleEngine.popen_uci(str(engine_path), env=_prepare_env()) as engine:
            engine.configure({"Hash": 64, "Threads": 4, "Model": "small", "DebugLog": False})

            for movetime_ms in movetimes_ms:
                t0 = time.perf_counter()
                result = engine.play(
                    board,
                    chess.engine.Limit(time=movetime_ms / 1000.0),
                    info=chess.engine.INFO_ALL,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                tolerance_ms = max(15.0, movetime_ms * 0.08)
                upper_bound = movetime_ms + tolerance_ms

                self.assertLessEqual(
                    elapsed_ms,
                    upper_bound,
                    f"movetime {movetime_ms}ms exceeded: actual={elapsed_ms:.1f}ms upper={upper_bound:.1f}ms",
                )
                self.assertTrue(board.is_legal(result.move), f"engine returned illegal move: {result.move}")


if __name__ == "__main__":
    unittest.main()
