#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import random
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

_CHESS_IMPORT_ERROR: Exception | None = None

try:
    import chess
    import chess.engine
except Exception as exc:  # pragma: no cover - exercised by dependency checks
    chess = None  # type: ignore[assignment]
    _CHESS_IMPORT_ERROR = exc


DEFAULT_MODELS = ["small", "large", "hybrid_root"]
DEFAULT_SF_ELOS = [1200, 1400, 1600, 1800, 2000]
MIN_SCORE_FOR_ELO = 0.01
MAX_SCORE_FOR_ELO = 0.99
CI_Z = 1.96
MIN_ANCHOR_STD_ELO = 1.0
DEFAULT_STOCKFISH_ELO_MIN = 1200
DEFAULT_STOCKFISH_ELO_MAX = 3200


@dataclass(frozen=True)
class Opening:
    id: str
    fen: str


@dataclass(frozen=True)
class ScheduledGame:
    game_id: str
    seed_token: str
    model: str
    sf_elo: int
    pair_index: int
    opening: Opening
    brainstorm_color: str


class EloEstimatorError(RuntimeError):
    pass


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


def require_python_chess() -> None:
    if chess is not None:
        return

    detail = f": {_CHESS_IMPORT_ERROR}" if _CHESS_IMPORT_ERROR else ""
    raise EloEstimatorError(
        "python-chess is required for Elo estimation. Install with `pip install python-chess`"
        f"{detail}"
    )


def parse_csv_items(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return items


def parse_models(value: str) -> List[str]:
    return parse_csv_items(value)


def parse_sf_elos(value: str) -> List[int]:
    items = parse_csv_items(value)
    parsed: List[int] = []
    for item in items:
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid Elo value: {item}") from exc
    return parsed


def resolve_engine_path(path_or_name: str, label: str) -> str:
    candidate = pathlib.Path(path_or_name).expanduser()

    resolved: pathlib.Path | None = None
    if candidate.is_file():
        resolved = candidate.resolve()
    elif any(sep in path_or_name for sep in ("/", "\\")) or path_or_name.startswith("."):
        resolved = None
    else:
        which = shutil.which(path_or_name)
        if which:
            resolved = pathlib.Path(which).resolve()

    if resolved is None:
        raise EloEstimatorError(f"{label} binary not found: {path_or_name}")

    if not os.access(resolved, os.X_OK):
        raise EloEstimatorError(f"{label} binary is not executable: {resolved}")

    return str(resolved)


def load_openings(path: pathlib.Path) -> List[Opening]:
    if not path.exists():
        raise EloEstimatorError(f"openings file not found: {path}")

    openings: List[Opening] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EloEstimatorError(f"invalid JSON in openings file at line {line_number}") from exc

            opening_id = str(payload.get("id", "")).strip()
            fen = str(payload.get("fen", "")).strip()
            if not opening_id or not fen:
                raise EloEstimatorError(f"openings line {line_number} missing `id` or `fen`")

            try:
                chess.Board(fen)  # type: ignore[union-attr]
            except Exception as exc:
                raise EloEstimatorError(
                    f"invalid FEN in openings file at line {line_number}: {fen}"
                ) from exc

            openings.append(Opening(id=opening_id, fen=fen))

    if not openings:
        raise EloEstimatorError("openings file is empty")

    return openings


def pick_openings(openings: Sequence[Opening], pairs_per_elo: int, seed_token: str) -> List[Opening]:
    if pairs_per_elo < 1:
        raise EloEstimatorError("pairs-per-elo must be >= 1")

    rng = random.Random(seed_token)
    if pairs_per_elo <= len(openings):
        return rng.sample(list(openings), pairs_per_elo)

    return [rng.choice(openings) for _ in range(pairs_per_elo)]


def build_schedule(
    models: Sequence[str],
    sf_elos: Sequence[int],
    openings: Sequence[Opening],
    pairs_per_elo: int,
    base_seed: int,
) -> List[ScheduledGame]:
    games: List[ScheduledGame] = []
    for model in models:
        for sf_elo in sf_elos:
            selection_seed = f"{base_seed}:{model}:{sf_elo}:openings"
            chosen = pick_openings(openings, pairs_per_elo, selection_seed)
            for pair_index, opening in enumerate(chosen):
                for brainstorm_color in ("white", "black"):
                    seed_token = (
                        f"{base_seed}:{model}:{sf_elo}:{opening.id}:{pair_index}:{brainstorm_color}"
                    )
                    games.append(
                        ScheduledGame(
                            game_id=seed_token,
                            seed_token=seed_token,
                            model=model,
                            sf_elo=sf_elo,
                            pair_index=pair_index,
                            opening=opening,
                            brainstorm_color=brainstorm_color,
                        )
                    )
    return games


def parse_existing_game_ids(games_path: pathlib.Path) -> set[str]:
    if not games_path.exists():
        return set()

    existing: set[str] = set()
    with games_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            game_id = payload.get("game_id")
            if isinstance(game_id, str) and game_id:
                existing.add(game_id)
    return existing


def load_all_game_records(games_path: pathlib.Path) -> List[dict]:
    records: List[dict] = []
    if not games_path.exists():
        return records

    with games_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def game_result_for_brainstorm(result: str, brainstorm_color: str) -> Tuple[float, str]:
    if result == "1/2-1/2":
        return 0.5, "draw"

    if result == "1-0":
        return (1.0, "win") if brainstorm_color == "white" else (0.0, "loss")

    if result == "0-1":
        return (1.0, "win") if brainstorm_color == "black" else (0.0, "loss")

    raise EloEstimatorError(f"unsupported result string: {result}")


def outcome_termination_name(outcome: chess.Outcome | None) -> str:  # type: ignore[name-defined]
    if outcome is None or outcome.termination is None:
        return "unknown"
    return str(outcome.termination).split(".")[-1].lower()


def elo_diff_from_score(score: float) -> float:
    clipped = min(MAX_SCORE_FOR_ELO, max(MIN_SCORE_FOR_ELO, score))
    return -400.0 * math.log10((1.0 / clipped) - 1.0)


def score_sample_variance(scores: Sequence[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    return statistics.variance(scores)


def score_standard_error(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    var = score_sample_variance(scores)
    return math.sqrt(var / len(scores))


def score_standard_error_with_halfpoint_floor(
    scores: Sequence[float], wins: int, draws: int, losses: int
) -> float:
    # Sample variance can collapse to 0 with tiny samples at the boundary (all wins/losses),
    # which yields unrealistically narrow Elo CIs. Add a conservative floor based on
    # half-points (2*games Bernoulli approximation with Laplace smoothing).
    sample_se = score_standard_error(scores)
    games = wins + draws + losses
    if games <= 0:
        return sample_se

    half_trials = 2 * games
    half_points = (2 * wins) + draws
    p_laplace = (half_points + 1.0) / (half_trials + 2.0)
    halfpoint_se = math.sqrt((p_laplace * (1.0 - p_laplace)) / half_trials)
    return max(sample_se, halfpoint_se)


def score_confidence_interval(score: float, standard_error: float, z: float = CI_Z) -> Tuple[float, float]:
    lo = max(MIN_SCORE_FOR_ELO, score - z * standard_error)
    hi = min(MAX_SCORE_FOR_ELO, score + z * standard_error)
    if hi < lo:
        hi = lo
    return lo, hi


def elo_std_from_score(score: float, score_se: float) -> float:
    clipped_score = min(MAX_SCORE_FOR_ELO, max(MIN_SCORE_FOR_ELO, score))
    derivative = 400.0 / (math.log(10.0) * clipped_score * (1.0 - clipped_score))
    std = abs(derivative * score_se)
    return max(MIN_ANCHOR_STD_ELO, std)


def combine_anchor_estimates(anchors: Sequence[dict]) -> dict:
    if not anchors:
        return {
            "elo": None,
            "std_elo": None,
            "ci95": [None, None],
            "weights": [],
        }

    weights: List[float] = []
    weighted_sum = 0.0
    for anchor in anchors:
        estimate = float(anchor["elo_estimate"])
        std_elo = max(float(anchor["elo_std"]), MIN_ANCHOR_STD_ELO)
        variance = std_elo * std_elo
        weight = 1.0 / variance
        weights.append(weight)
        weighted_sum += estimate * weight

    total_weight = sum(weights)
    combined = weighted_sum / total_weight
    combined_std = math.sqrt(1.0 / total_weight)
    ci_lo = combined - CI_Z * combined_std
    ci_hi = combined + CI_Z * combined_std

    return {
        "elo": round(combined, 2),
        "std_elo": round(combined_std, 2),
        "ci95": [round(ci_lo, 2), round(ci_hi, 2)],
        "weights": [round(w, 8) for w in weights],
    }


def group_games(games: Sequence[ScheduledGame]) -> Dict[Tuple[str, int], List[ScheduledGame]]:
    grouped: Dict[Tuple[str, int], List[ScheduledGame]] = {}
    for game in games:
        key = (game.model, game.sf_elo)
        grouped.setdefault(key, []).append(game)
    return grouped


def group_games_by_model_then_elo(
    games: Sequence[ScheduledGame],
) -> Dict[str, Dict[int, List[ScheduledGame]]]:
    grouped: Dict[str, Dict[int, List[ScheduledGame]]] = {}
    for game in games:
        grouped.setdefault(game.model, {}).setdefault(game.sf_elo, []).append(game)
    return grouped


def validate_stockfish_options(engine: chess.engine.SimpleEngine) -> None:  # type: ignore[name-defined]
    needed = ["UCI_LimitStrength", "UCI_Elo"]
    missing = [name for name in needed if name not in engine.options]
    if missing:
        missing_str = ", ".join(missing)
        raise EloEstimatorError(
            f"Stockfish does not expose required UCI options: {missing_str}. "
            "Use a Stockfish build that supports UCI_Elo."
        )


def _as_int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def detect_stockfish_elo_bounds(stockfish_path: str) -> Tuple[int, int]:
    with chess.engine.SimpleEngine.popen_uci(stockfish_path, env=_prepare_env()) as stockfish_engine:  # type: ignore[union-attr]
        validate_stockfish_options(stockfish_engine)
        option = stockfish_engine.options["UCI_Elo"]
        min_elo = _as_int_or_none(getattr(option, "min", None))
        max_elo = _as_int_or_none(getattr(option, "max", None))

    if min_elo is None:
        min_elo = DEFAULT_STOCKFISH_ELO_MIN
    if max_elo is None:
        max_elo = DEFAULT_STOCKFISH_ELO_MAX
    if min_elo > max_elo:
        raise EloEstimatorError(
            f"Stockfish reported invalid UCI_Elo bounds: min={min_elo}, max={max_elo}"
        )
    return min_elo, max_elo


def normalize_sf_elos(
    requested: Sequence[int], min_elo: int, max_elo: int
) -> Tuple[List[int], List[Tuple[int, int]], List[int]]:
    normalized: List[int] = []
    clipped: List[Tuple[int, int]] = []
    for value in requested:
        adjusted = min(max(value, min_elo), max_elo)
        if adjusted != value:
            clipped.append((value, adjusted))
        normalized.append(adjusted)

    unique: List[int] = []
    dropped_duplicates: List[int] = []
    seen: set[int] = set()
    for value in normalized:
        if value in seen:
            dropped_duplicates.append(value)
            continue
        seen.add(value)
        unique.append(value)

    return unique, clipped, dropped_duplicates


def play_single_game(
    game: ScheduledGame,
    brainstorm_engine: chess.engine.SimpleEngine,  # type: ignore[name-defined]
    stockfish_engine: chess.engine.SimpleEngine,  # type: ignore[name-defined]
    limit: chess.engine.Limit,  # type: ignore[name-defined]
    max_plies: int,
) -> dict:
    board = chess.Board(game.opening.fen)  # type: ignore[union-attr]
    plies = 0
    t0 = time.perf_counter()

    result = "1/2-1/2"
    termination = "unknown"

    while True:
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                result = outcome.result()
                termination = outcome_termination_name(outcome)
            else:
                termination = "game_over"
            break

        if plies >= max_plies:
            result = "1/2-1/2"
            termination = "max_plies"
            break

        turn_color = "white" if board.turn == chess.WHITE else "black"  # type: ignore[union-attr]
        brainstorm_to_move = turn_color == game.brainstorm_color
        engine = brainstorm_engine if brainstorm_to_move else stockfish_engine

        move_result = engine.play(board, limit, info=chess.engine.INFO_NONE)  # type: ignore[union-attr]
        move = move_result.move
        if move is None or move not in board.legal_moves:
            result = "0-1" if board.turn == chess.WHITE else "1-0"  # type: ignore[union-attr]
            termination = "illegal_or_missing_move"
            break

        board.push(move)
        plies += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    score, verdict = game_result_for_brainstorm(result, game.brainstorm_color)

    return {
        "game_id": game.game_id,
        "seed_token": game.seed_token,
        "model": game.model,
        "sf_elo": game.sf_elo,
        "opening_id": game.opening.id,
        "opening_fen": game.opening.fen,
        "pair_index": game.pair_index,
        "color": game.brainstorm_color,
        "brainstorm_color": game.brainstorm_color,
        "result": result,
        "brainstorm_result": verdict,
        "score": score,
        "plies": plies,
        "termination": termination,
        "wall_clock_ms": round(elapsed_ms, 2),
        "played_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def play_matches(
    games: Sequence[ScheduledGame],
    brainstorm_path: str,
    stockfish_path: str,
    movetime_ms: int,
    threads: int,
    hash_mb: int,
    brainstorm_device: str,
    max_plies: int,
    games_file: pathlib.Path,
) -> None:
    grouped = group_games_by_model_then_elo(games)
    limit = chess.engine.Limit(time=movetime_ms / 1000.0)  # type: ignore[union-attr]

    total = len(games)
    played = 0

    with games_file.open("a", encoding="utf-8") as handle:
        for model, elo_groups in grouped.items():
            total_model_games = sum(len(batch) for batch in elo_groups.values())
            print(f"[elo] model={model} games={total_model_games}", file=sys.stderr)

            with chess.engine.SimpleEngine.popen_uci(brainstorm_path, env=_prepare_env()) as brainstorm_engine:  # type: ignore[union-attr]
                brainstorm_config = {
                    "Hash": hash_mb,
                    "Threads": threads,
                    "Model": model,
                    "DebugLog": False,
                }
                if "Device" in brainstorm_engine.options:
                    brainstorm_config["Device"] = brainstorm_device
                elif brainstorm_device != "auto":
                    print(
                        "[elo] warning: Brainstorm engine has no UCI `Device` option; "
                        f"ignoring --brainstorm-device={brainstorm_device}",
                        file=sys.stderr,
                    )

                brainstorm_engine.configure(brainstorm_config)

                with chess.engine.SimpleEngine.popen_uci(stockfish_path, env=_prepare_env()) as stockfish_engine:  # type: ignore[union-attr]
                    validate_stockfish_options(stockfish_engine)
                    stockfish_engine.configure(
                        {
                            "Hash": hash_mb,
                            "Threads": threads,
                            "UCI_LimitStrength": True,
                        }
                    )

                    for sf_elo in sorted(elo_groups.keys()):
                        batch = elo_groups[sf_elo]
                        print(
                            f"[elo] batch model={model} sf_elo={sf_elo} games={len(batch)}",
                            file=sys.stderr,
                        )
                        stockfish_engine.configure({"UCI_Elo": sf_elo})

                        for game in batch:
                            record = play_single_game(
                                game=game,
                                brainstorm_engine=brainstorm_engine,
                                stockfish_engine=stockfish_engine,
                                limit=limit,
                                max_plies=max_plies,
                            )
                            handle.write(json.dumps(record) + "\n")
                            handle.flush()
                            played += 1
                            print(
                                f"[elo] progress {played}/{total} game_id={game.game_id}",
                                file=sys.stderr,
                            )


def summarize_records(records: Sequence[dict], models: Sequence[str], sf_elos: Sequence[int]) -> dict:
    by_model: Dict[str, dict] = {}

    for model in models:
        model_records = [item for item in records if item.get("model") == model]
        anchors: List[dict] = []

        for sf_elo in sf_elos:
            anchor_records = [
                item for item in model_records if int(item.get("sf_elo", -1)) == int(sf_elo)
            ]
            if not anchor_records:
                continue

            scores = [float(item["score"]) for item in anchor_records]
            wins = sum(1 for item in anchor_records if item.get("brainstorm_result") == "win")
            draws = sum(1 for item in anchor_records if item.get("brainstorm_result") == "draw")
            losses = sum(1 for item in anchor_records if item.get("brainstorm_result") == "loss")

            mean_score = statistics.mean(scores)
            score_var = score_sample_variance(scores)
            score_se = score_standard_error_with_halfpoint_floor(
                scores=scores,
                wins=wins,
                draws=draws,
                losses=losses,
            )

            elo_diff = elo_diff_from_score(mean_score)
            elo_estimate = sf_elo + elo_diff

            score_ci_lo, score_ci_hi = score_confidence_interval(mean_score, score_se)
            elo_ci_lo = sf_elo + elo_diff_from_score(score_ci_lo)
            elo_ci_hi = sf_elo + elo_diff_from_score(score_ci_hi)

            elo_std = elo_std_from_score(mean_score, score_se)

            anchors.append(
                {
                    "sf_elo": sf_elo,
                    "games": len(anchor_records),
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "mean_score": round(mean_score, 4),
                    "score_variance": round(score_var, 6),
                    "score_se": round(score_se, 6),
                    "score_ci95": [round(score_ci_lo, 4), round(score_ci_hi, 4)],
                    "elo_diff": round(elo_diff, 2),
                    "elo_estimate": round(elo_estimate, 2),
                    "elo_ci95": [round(elo_ci_lo, 2), round(elo_ci_hi, 2)],
                    "elo_std": round(elo_std, 2),
                }
            )

        combined = combine_anchor_estimates(anchors)
        total_games = len(model_records)
        total_wins = sum(1 for item in model_records if item.get("brainstorm_result") == "win")
        total_draws = sum(1 for item in model_records if item.get("brainstorm_result") == "draw")
        total_losses = sum(1 for item in model_records if item.get("brainstorm_result") == "loss")
        aggregate_score = (
            sum(float(item.get("score", 0.0)) for item in model_records) / total_games
            if total_games
            else 0.0
        )

        by_model[model] = {
            "combined": combined,
            "anchors": anchors,
            "aggregate": {
                "games": total_games,
                "wins": total_wins,
                "draws": total_draws,
                "losses": total_losses,
                "mean_score": round(aggregate_score, 4),
            },
        }

    return by_model


def print_console_summary(summary_by_model: dict) -> None:
    headers = ["Model", "Games", "W", "D", "L", "Score%", "Elo", "CI95"]
    print("\n" + " | ".join(headers))
    print("-" * 74)

    for model, payload in summary_by_model.items():
        aggregate = payload["aggregate"]
        combined = payload["combined"]

        elo = combined["elo"]
        ci_lo, ci_hi = combined["ci95"]
        ci_text = "n/a" if ci_lo is None else f"[{ci_lo:.2f}, {ci_hi:.2f}]"
        elo_text = "n/a" if elo is None else f"{elo:.2f}"

        print(
            " | ".join(
                [
                    f"{model}",
                    str(aggregate["games"]),
                    str(aggregate["wins"]),
                    str(aggregate["draws"]),
                    str(aggregate["losses"]),
                    f"{aggregate['mean_score'] * 100:.1f}",
                    elo_text,
                    ci_text,
                ]
            )
        )


def resolve_run_dir(output_dir: pathlib.Path, resume: bool) -> pathlib.Path:
    output_dir = output_dir.resolve()

    if resume:
        direct_games = output_dir / "games.jsonl"
        if direct_games.exists():
            return output_dir

        if output_dir.is_dir():
            candidates = sorted(
                [
                    entry
                    for entry in output_dir.iterdir()
                    if entry.is_dir() and (entry / "games.jsonl").exists()
                ]
            )
            if candidates:
                return candidates[-1]

        raise EloEstimatorError(
            f"resume requested but no run with games.jsonl found under: {output_dir}"
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate Brainstorm Elo by running UCI matches against Stockfish UCI_Elo anchors"
    )
    parser.add_argument("--brainstorm", default="./brainstorm", help="Brainstorm engine binary path")
    parser.add_argument("--stockfish", default="stockfish", help="Stockfish binary path or name in PATH")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        type=parse_models,
        help="Comma-separated Brainstorm model modes",
    )
    parser.add_argument(
        "--sf-elos",
        default=",".join(str(v) for v in DEFAULT_SF_ELOS),
        type=parse_sf_elos,
        help="Comma-separated Stockfish UCI_Elo anchors",
    )
    parser.add_argument("--pairs-per-elo", type=int, default=12, help="Opening pairs per Stockfish Elo")
    parser.add_argument("--movetime-ms", type=int, default=200, help="Fixed movetime for both engines")
    parser.add_argument("--threads", type=int, default=1, help="Threads for both engines")
    parser.add_argument("--hash-mb", type=int, default=64, help="Hash MB for both engines")
    parser.add_argument(
        "--brainstorm-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Brainstorm evaluator device preference (requires engine support for UCI Device)",
    )
    parser.add_argument(
        "--openings",
        type=pathlib.Path,
        default=pathlib.Path("scripts/elo_openings.jsonl"),
        help="JSONL opening suite path",
    )
    parser.add_argument("--max-plies", type=int, default=240, help="Force draw at this ply count")
    parser.add_argument("--seed", type=int, default=42, help="Base deterministic seed")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("results/elo"),
        help="Output directory (new timestamped run folder is created unless --resume)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing games.jsonl under --output-dir",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    require_python_chess()

    if args.pairs_per_elo < 1:
        raise EloEstimatorError("--pairs-per-elo must be >= 1")
    if args.movetime_ms < 1:
        raise EloEstimatorError("--movetime-ms must be >= 1")
    if args.threads < 1:
        raise EloEstimatorError("--threads must be >= 1")
    if args.hash_mb < 1:
        raise EloEstimatorError("--hash-mb must be >= 1")
    if args.max_plies < 1:
        raise EloEstimatorError("--max-plies must be >= 1")

    brainstorm_path = resolve_engine_path(args.brainstorm, "Brainstorm")
    stockfish_path = resolve_engine_path(args.stockfish, "Stockfish")
    sf_elo_min, sf_elo_max = detect_stockfish_elo_bounds(stockfish_path)
    effective_sf_elos, clipped_elos, duplicate_elos = normalize_sf_elos(
        requested=args.sf_elos,
        min_elo=sf_elo_min,
        max_elo=sf_elo_max,
    )
    if not effective_sf_elos:
        raise EloEstimatorError("no usable Stockfish Elo anchors after normalization")

    run_dir = resolve_run_dir(args.output_dir, resume=args.resume)
    run_dir.mkdir(parents=True, exist_ok=True)

    games_file = run_dir / "games.jsonl"
    summary_file = run_dir / "summary.json"

    openings = load_openings(args.openings)

    schedule = build_schedule(
        models=args.models,
        sf_elos=effective_sf_elos,
        openings=openings,
        pairs_per_elo=args.pairs_per_elo,
        base_seed=args.seed,
    )

    existing_ids = parse_existing_game_ids(games_file)
    pending = [game for game in schedule if game.game_id not in existing_ids]

    print(f"[elo] run directory: {run_dir}", file=sys.stderr)
    print(f"[elo] scheduled games: {len(schedule)}", file=sys.stderr)
    print(f"[elo] existing games: {len(existing_ids)}", file=sys.stderr)
    print(f"[elo] pending games: {len(pending)}", file=sys.stderr)
    print(f"[elo] stockfish UCI_Elo range: [{sf_elo_min}, {sf_elo_max}]", file=sys.stderr)
    if clipped_elos:
        for requested, adjusted in clipped_elos:
            print(
                f"[elo] warning: requested sf_elo={requested} is outside "
                f"[{sf_elo_min}, {sf_elo_max}], using {adjusted}",
                file=sys.stderr,
            )
    if duplicate_elos:
        unique_dups = sorted(set(duplicate_elos))
        print(
            f"[elo] warning: duplicate sf_elos after normalization were dropped: {unique_dups}",
            file=sys.stderr,
        )

    if pending:
        play_matches(
            games=pending,
            brainstorm_path=brainstorm_path,
            stockfish_path=stockfish_path,
            movetime_ms=args.movetime_ms,
            threads=args.threads,
            hash_mb=args.hash_mb,
            brainstorm_device=args.brainstorm_device,
            max_plies=args.max_plies,
            games_file=games_file,
        )

    records = load_all_game_records(games_file)
    summary_by_model = summarize_records(records, models=args.models, sf_elos=effective_sf_elos)

    payload = {
        "run": {
            "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "run_dir": str(run_dir),
            "brainstorm": brainstorm_path,
            "stockfish": stockfish_path,
            "models": args.models,
            "requested_sf_elos": args.sf_elos,
            "sf_elos": effective_sf_elos,
            "sf_elo_range": [sf_elo_min, sf_elo_max],
            "pairs_per_elo": args.pairs_per_elo,
            "movetime_ms": args.movetime_ms,
            "threads": args.threads,
            "hash_mb": args.hash_mb,
            "brainstorm_device": args.brainstorm_device,
            "openings": str(args.openings.resolve()),
            "max_plies": args.max_plies,
            "seed": args.seed,
            "resume": args.resume,
            "scheduled_games": len(schedule),
            "completed_games": len(records),
        },
        "models": summary_by_model,
    }

    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print_console_summary(summary_by_model)
    print(f"\n[elo] wrote {games_file}")
    print(f"[elo] wrote {summary_file}")

    return 0


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        code = run(args)
    except EloEstimatorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        raise SystemExit(130)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
