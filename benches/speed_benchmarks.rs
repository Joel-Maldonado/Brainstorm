use brainstorm::search_algorithm::{ModelMode, SearchAlgorithm, SearchOptions, SearchRequest};
use brainstorm::utils::{
    board_to_tensor, encode_board_features, order_captures, order_moves, HistoryTable,
};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use pleco::core::GenTypes;
use pleco::{BitMove, Board};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tch::{CModule, Device};

const SMALL_MODEL_PATH: &str = "models/eval_params264k_norm_mse0.117666_jit.pt";
const LARGE_MODEL_PATH: &str = "models/eval_660k_norm_mse_0.026550_jit.pt";

const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const MIDDLEGAME_FEN: &str = "r2q1rk1/pp1b1ppp/2n1pn2/2bp4/2P5/2NP1NP1/PP2PPBP/R1BQ1RK1 w - - 0 8";
const TACTICAL_FEN: &str = "2r2rk1/pp1n1pp1/2p1pn1p/2Pp4/3P4/2N1PN2/PP3PPP/2RR2K1 w - - 0 15";
const ENDGAME_FEN: &str = "8/5pk1/3p1np1/2pPp3/2P1P3/3N1P2/5K1P/8 w - - 0 40";

struct BenchModels {
    small: Arc<CModule>,
    large: Arc<CModule>,
}

static MODELS: OnceLock<BenchModels> = OnceLock::new();

fn board_from_fen(fen: &str) -> Board {
    Board::from_fen(fen).expect("benchmark FEN must be valid")
}

fn bench_positions() -> Vec<(&'static str, Board)> {
    vec![
        ("startpos", board_from_fen(STARTPOS_FEN)),
        ("middlegame", board_from_fen(MIDDLEGAME_FEN)),
        ("tactical", board_from_fen(TACTICAL_FEN)),
        ("endgame", board_from_fen(ENDGAME_FEN)),
    ]
}

fn load_models() -> &'static BenchModels {
    MODELS.get_or_init(|| {
        let small = CModule::load_on_device(SMALL_MODEL_PATH, Device::Cpu)
            .expect("failed to load small model for benchmarks");
        let large = CModule::load_on_device(LARGE_MODEL_PATH, Device::Cpu)
            .expect("failed to load large model for benchmarks");
        BenchModels {
            small: Arc::new(small),
            large: Arc::new(large),
        }
    })
}

fn benchmark_feature_encoding(c: &mut Criterion) {
    let positions = bench_positions();
    let mut group = c.benchmark_group("feature_encoding");

    for (name, board) in &positions {
        group.bench_with_input(BenchmarkId::from_parameter(name), board, |b, board| {
            let mut features = [0_f32; 775];
            b.iter(|| encode_board_features(board, &mut features));
        });
    }

    group.finish();
}

fn benchmark_move_ordering(c: &mut Criterion) {
    let positions = bench_positions();
    let mut group = c.benchmark_group("move_ordering");
    let history: HistoryTable = [[[0; 64]; 64]; 2];
    let killers = [BitMove::null(), BitMove::null()];

    for (name, board) in &positions {
        group.bench_with_input(BenchmarkId::new("all_moves", name), board, |b, board| {
            b.iter_batched(
                || board.generate_moves().to_vec(),
                |mut moves| {
                    order_moves(board, &mut moves, None, killers, &history, board.turn());
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("captures", name), board, |b, board| {
            b.iter_batched(
                || board.generate_moves_of_type(GenTypes::Captures).to_vec(),
                |mut captures| order_captures(board, &mut captures),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn benchmark_model_inference(c: &mut Criterion) {
    let positions = bench_positions();
    let models = load_models();
    let mut group = c.benchmark_group("model_inference_cpu");
    group.sample_size(15);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    for (name, board) in &positions {
        group.bench_with_input(
            BenchmarkId::new("small_pipeline", name),
            board,
            |b, board| {
                b.iter(|| {
                    let input = board_to_tensor(board).to_device(Device::Cpu);
                    let _ = models
                        .small
                        .forward_ts(&[input])
                        .expect("small model forward should succeed");
                });
            },
        );
    }

    for (name, board) in &positions {
        group.bench_with_input(
            BenchmarkId::new("large_pipeline", name),
            board,
            |b, board| {
                b.iter(|| {
                    let input = board_to_tensor(board).to_device(Device::Cpu);
                    let _ = models
                        .large
                        .forward_ts(&[input])
                        .expect("large model forward should succeed");
                });
            },
        );
    }

    let reference_input = board_to_tensor(&positions[1].1).to_device(Device::Cpu);
    group.bench_function("small_forward_only", |b| {
        b.iter(|| {
            let _ = models
                .small
                .forward_ts(&[reference_input.shallow_clone()])
                .expect("small model forward should succeed");
        });
    });
    group.bench_function("large_forward_only", |b| {
        b.iter(|| {
            let _ = models
                .large
                .forward_ts(&[reference_input.shallow_clone()])
                .expect("large model forward should succeed");
        });
    });

    group.finish();
}

fn benchmark_search(c: &mut Criterion) {
    let models = load_models();
    let searcher = SearchAlgorithm::new(
        Arc::clone(&models.small),
        Arc::clone(&models.large),
        Device::Cpu,
        Arc::new(AtomicBool::new(false)),
    );

    let mut group = c.benchmark_group("search_cpu");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(10));

    let cases: [(&str, ModelMode, u32, &str); 4] = [
        ("small_d2_startpos", ModelMode::Small, 2, STARTPOS_FEN),
        ("small_d3_middlegame", ModelMode::Small, 3, MIDDLEGAME_FEN),
        ("large_d2_middlegame", ModelMode::Large, 2, MIDDLEGAME_FEN),
        ("hybrid_d3_tactical", ModelMode::HybridRoot, 3, TACTICAL_FEN),
    ];

    for (name, model_mode, depth, fen) in cases {
        let board = board_from_fen(fen);
        let history = vec![board.zobrist()];
        let request = SearchRequest {
            max_depth: depth,
            soft_time_ms: None,
            hard_time_ms: None,
        };
        let options = SearchOptions {
            hash_mb: 64,
            threads: 1,
            model_mode,
            debug_log: false,
        };

        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                let _ = searcher.search_quiet(&board, request, &options, &history);
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = speed_benches;
    config = Criterion::default().warm_up_time(Duration::from_secs(2));
    targets =
        benchmark_feature_encoding,
        benchmark_move_ordering,
        benchmark_model_inference,
        benchmark_search
);
criterion_main!(speed_benches);
