from __future__ import annotations

import logging
import os
import pathlib
import unittest

import chess
import chess.engine


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


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


class UCIProtocolTest(unittest.TestCase):
    def test_no_unexpected_output_and_legal_move(self) -> None:
        engine_path = pathlib.Path(__file__).resolve().parents[1] / "brainstorm"
        if not engine_path.exists():
            self.skipTest("engine binary not found")

        capture = _LogCapture()
        logger = logging.getLogger("chess.engine")
        old_level = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(capture)

        try:
            with chess.engine.SimpleEngine.popen_uci(str(engine_path), env=_prepare_env()) as engine:
                engine.configure({"Hash": 64, "Threads": 1, "Model": "fast", "DebugLog": False})

                board = chess.Board()
                result = engine.play(board, chess.engine.Limit(depth=3), info=chess.engine.INFO_ALL)
                self.assertTrue(board.is_legal(result.move), f"engine returned illegal move: {result.move}")

                board = chess.Board(
                    "r2q1rk1/pp1b1ppp/2n1pn2/2bp4/2P5/2NP1NP1/PP2PPBP/R1BQ1RK1 w - - 0 8"
                )
                result = engine.play(board, chess.engine.Limit(depth=4), info=chess.engine.INFO_ALL)
                self.assertTrue(board.is_legal(result.move), f"engine returned illegal move: {result.move}")
        finally:
            logger.removeHandler(capture)
            logger.setLevel(old_level)

        unexpected = [
            record.getMessage() for record in capture.records if "Unexpected engine output" in record.getMessage()
        ]
        self.assertEqual([], unexpected, f"unexpected engine output detected: {unexpected}")


if __name__ == "__main__":
    unittest.main()
