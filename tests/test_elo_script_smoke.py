from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest


def _has_python_chess() -> bool:
    return importlib.util.find_spec("chess") is not None


def _write_fake_uci_engine(
    path: pathlib.Path,
    role: str,
    expose_uci_elo: bool = True,
    uci_elo_min: int = 1200,
    uci_elo_max: int = 3200,
) -> None:
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        from __future__ import annotations

        import chess
        import sys

        ROLE = {role!r}
        EXPOSE_UCI_ELO = {expose_uci_elo!r}
        UCI_ELO_MIN = {uci_elo_min!r}
        UCI_ELO_MAX = {uci_elo_max!r}
        board = chess.Board()
        options = {{}}


        def parse_option(command: str):
            tokens = command.strip().split()
            if not tokens or tokens[0] != "setoption":
                return None, None

            name_parts = []
            value_parts = []
            in_name = False
            in_value = False

            for token in tokens[1:]:
                if token == "name":
                    in_name = True
                    in_value = False
                    continue
                if token == "value":
                    in_name = False
                    in_value = True
                    continue
                if in_name:
                    name_parts.append(token)
                elif in_value:
                    value_parts.append(token)

            name = " ".join(name_parts)
            value = " ".join(value_parts)
            return name, value


        def set_position(command: str):
            global board
            tokens = command.strip().split()
            if len(tokens) < 2:
                return

            if tokens[1] == "startpos":
                board = chess.Board()
                idx = 2
            elif tokens[1] == "fen":
                idx = 2
                fen_parts = []
                while idx < len(tokens) and tokens[idx] != "moves":
                    fen_parts.append(tokens[idx])
                    idx += 1
                try:
                    board = chess.Board(" ".join(fen_parts))
                except Exception:
                    board = chess.Board()
            else:
                return

            if idx < len(tokens) and tokens[idx] == "moves":
                idx += 1
                for mv in tokens[idx:]:
                    try:
                        board.push_uci(mv)
                    except Exception:
                        break


        def pick_move() -> str:
            legal = sorted(m.uci() for m in board.legal_moves)
            if not legal:
                return "0000"

            if ROLE == "stockfish":
                elo = int(options.get("UCI_Elo", "1500") or "1500")
                if elo >= 1700:
                    return legal[0]
                return legal[-1]

            # Brainstorm stub: deterministic middle-ish move.
            return legal[len(legal) // 2]


        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue

            cmd = line.split()[0]

            if cmd == "uci":
                print(f"id name Fake{{ROLE.capitalize()}}", flush=True)
                print("id author test", flush=True)
                print("option name Hash type spin default 16 min 1 max 1024", flush=True)
                print("option name Threads type spin default 1 min 1 max 8", flush=True)

                if ROLE == "brainstorm":
                    print("option name Model type combo default fast var fast var balanced var accurate", flush=True)
                    print("option name DebugLog type check default false", flush=True)
                else:
                    print("option name UCI_LimitStrength type check default false", flush=True)
                    if EXPOSE_UCI_ELO:
                        print(
                            f"option name UCI_Elo type spin default 1500 min {{UCI_ELO_MIN}} max {{UCI_ELO_MAX}}",
                            flush=True,
                        )

                print("uciok", flush=True)
            elif cmd == "isready":
                print("readyok", flush=True)
            elif cmd == "setoption":
                name, value = parse_option(line)
                if name:
                    options[name] = value
            elif cmd == "ucinewgame":
                board = chess.Board()
            elif cmd == "position":
                set_position(line)
            elif cmd == "go":
                print(f"bestmove {{pick_move()}}", flush=True)
            elif cmd == "quit":
                break
        """
    )

    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_openings(path: pathlib.Path) -> None:
    lines = [
        {"id": "open_a", "fen": "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1"},
        {"id": "open_b", "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in lines:
            handle.write(json.dumps(row) + "\n")


@unittest.skipUnless(_has_python_chess(), "python-chess is required")
class EloScriptSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, "scripts/estimate_elo.py", *args]
        return subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True)

    def test_tiny_run_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = pathlib.Path(td)
            engines = temp / "engines"
            engines.mkdir(parents=True, exist_ok=True)

            brainstorm = engines / "fake_brainstorm.py"
            stockfish = engines / "fake_stockfish.py"
            _write_fake_uci_engine(brainstorm, role="brainstorm", expose_uci_elo=True)
            _write_fake_uci_engine(stockfish, role="stockfish", expose_uci_elo=True)

            openings = temp / "openings.jsonl"
            _write_openings(openings)

            out_dir = temp / "results"

            base_args = [
                "--brainstorm",
                str(brainstorm),
                "--stockfish",
                str(stockfish),
                "--models",
                "fast",
                "--sf-elos",
                "1200,1400",
                "--pairs-per-elo",
                "1",
                "--movetime-ms",
                "5",
                "--threads",
                "1",
                "--hash-mb",
                "16",
                "--openings",
                str(openings),
                "--max-plies",
                "8",
                "--seed",
                "123",
                "--output-dir",
                str(out_dir),
            ]

            first = self._run(base_args)
            self.assertEqual(first.returncode, 0, msg=first.stderr)

            run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]

            games_file = run_dir / "games.jsonl"
            summary_file = run_dir / "summary.json"
            self.assertTrue(games_file.exists())
            self.assertTrue(summary_file.exists())

            initial_lines = [line for line in games_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(initial_lines), 4)  # 1 model * 2 anchors * 1 pair * 2 colors

            second = self._run([*base_args, "--resume"])
            self.assertEqual(second.returncode, 0, msg=second.stderr)

            resumed_lines = [line for line in games_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(resumed_lines), len(initial_lines))

    def test_missing_stockfish_binary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = pathlib.Path(td)
            engines = temp / "engines"
            engines.mkdir(parents=True, exist_ok=True)

            brainstorm = engines / "fake_brainstorm.py"
            _write_fake_uci_engine(brainstorm, role="brainstorm", expose_uci_elo=True)

            openings = temp / "openings.jsonl"
            _write_openings(openings)

            result = self._run(
                [
                    "--brainstorm",
                    str(brainstorm),
                    "--stockfish",
                    str(temp / "does_not_exist_stockfish"),
                    "--models",
                    "fast",
                    "--sf-elos",
                    "1200",
                    "--pairs-per-elo",
                    "1",
                    "--openings",
                    str(openings),
                    "--output-dir",
                    str(temp / "results"),
                ]
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Stockfish binary not found", result.stderr)

    def test_missing_uci_elo_option(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = pathlib.Path(td)
            engines = temp / "engines"
            engines.mkdir(parents=True, exist_ok=True)

            brainstorm = engines / "fake_brainstorm.py"
            stockfish = engines / "fake_stockfish.py"
            _write_fake_uci_engine(brainstorm, role="brainstorm", expose_uci_elo=True)
            _write_fake_uci_engine(stockfish, role="stockfish", expose_uci_elo=False)

            openings = temp / "openings.jsonl"
            _write_openings(openings)

            result = self._run(
                [
                    "--brainstorm",
                    str(brainstorm),
                    "--stockfish",
                    str(stockfish),
                    "--models",
                    "fast",
                    "--sf-elos",
                    "1200",
                    "--pairs-per-elo",
                    "1",
                    "--movetime-ms",
                    "5",
                    "--openings",
                    str(openings),
                    "--output-dir",
                    str(temp / "results"),
                ]
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("UCI_Elo", result.stderr)

    def test_out_of_range_sf_elo_is_clipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = pathlib.Path(td)
            engines = temp / "engines"
            engines.mkdir(parents=True, exist_ok=True)

            brainstorm = engines / "fake_brainstorm.py"
            stockfish = engines / "fake_stockfish.py"
            _write_fake_uci_engine(brainstorm, role="brainstorm", expose_uci_elo=True)
            _write_fake_uci_engine(
                stockfish,
                role="stockfish",
                expose_uci_elo=True,
                uci_elo_min=1320,
                uci_elo_max=2200,
            )

            openings = temp / "openings.jsonl"
            _write_openings(openings)
            out_dir = temp / "results"

            result = self._run(
                [
                    "--brainstorm",
                    str(brainstorm),
                    "--stockfish",
                    str(stockfish),
                    "--models",
                    "fast",
                    "--sf-elos",
                    "1200,1400",
                    "--pairs-per-elo",
                    "1",
                    "--movetime-ms",
                    "5",
                    "--openings",
                    str(openings),
                    "--output-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("warning: requested sf_elo=1200", result.stderr)

            run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            summary_file = run_dirs[0] / "summary.json"
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            self.assertEqual(summary["run"]["requested_sf_elos"], [1200, 1400])
            self.assertEqual(summary["run"]["sf_elos"], [1320, 1400])
            self.assertEqual(summary["run"]["sf_elo_range"], [1320, 2200])


if __name__ == "__main__":
    unittest.main()
