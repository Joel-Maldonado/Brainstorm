use pleco::{BitMove, Board, Piece, PieceType, Player};
use tch::Tensor;

pub type HistoryTable = [[[i32; 64]; 64]; 2];

pub fn piece_to_offset(piece: Piece) -> usize {
    match piece {
        Piece::WhitePawn => 0,
        Piece::WhiteRook => 64,
        Piece::WhiteKnight => 128,
        Piece::WhiteBishop => 192,
        Piece::WhiteQueen => 256,
        Piece::WhiteKing => 320,
        Piece::BlackPawn => 384,
        Piece::BlackRook => 448,
        Piece::BlackKnight => 512,
        Piece::BlackBishop => 576,
        Piece::BlackQueen => 640,
        Piece::BlackKing => 704,
        Piece::None => unreachable!("board piece locations never contain Piece::None"),
    }
}

pub fn encode_board_features(board: &Board, features: &mut [f32; 775]) {
    features.fill(0.0);

    for (sq, piece) in board.get_piece_locations() {
        let sq_idx = (sq.file_idx_of_sq() + sq.rank_idx_of_sq() * 8) as usize;
        let idx = piece_to_offset(piece) + sq_idx;
        features[idx] = 1.0;
    }

    let turn = board.turn();
    features[768] = if turn == Player::White { 1.0 } else { 0.0 };

    if board.in_check() {
        if turn == Player::White {
            features[769] = 1.0;
        } else {
            features[770] = 1.0;
        }
    }

    features[771] =
        board.can_castle(Player::White, pleco::core::CastleType::KingSide) as i32 as f32;
    features[772] =
        board.can_castle(Player::White, pleco::core::CastleType::QueenSide) as i32 as f32;
    features[773] =
        board.can_castle(Player::Black, pleco::core::CastleType::KingSide) as i32 as f32;
    features[774] =
        board.can_castle(Player::Black, pleco::core::CastleType::QueenSide) as i32 as f32;
}

pub fn board_to_tensor(board: &Board) -> Tensor {
    let mut features = [0.0_f32; 775];
    encode_board_features(board, &mut features);
    Tensor::from_slice(&features).view([1, 775])
}

fn piece_value(piece_type: PieceType) -> i32 {
    match piece_type {
        PieceType::P => 100,
        PieceType::N => 320,
        PieceType::B => 330,
        PieceType::R => 500,
        PieceType::Q => 900,
        PieceType::K => 20_000,
        PieceType::None | PieceType::All => 0,
    }
}

fn mvv_lva_score(board: &Board, mv: BitMove) -> i32 {
    let victim_value = piece_value(board.piece_at_sq(mv.get_dest()).type_of());
    let attacker_value = piece_value(board.piece_at_sq(mv.get_src()).type_of());
    (victim_value * 10) - attacker_value
}

fn side_index(side: Player) -> usize {
    if side == Player::White {
        0
    } else {
        1
    }
}

fn move_score(
    board: &Board,
    mv: BitMove,
    tt_move: Option<BitMove>,
    killers: [BitMove; 2],
    history: &HistoryTable,
    side_to_move: Player,
) -> i32 {
    if Some(mv) == tt_move {
        return 2_000_000;
    }

    if board.is_capture_or_promotion(mv) {
        return 1_000_000 + mvv_lva_score(board, mv);
    }

    if mv == killers[0] {
        return 900_000;
    }
    if mv == killers[1] {
        return 899_000;
    }

    let from = mv.get_src_u8() as usize;
    let to = mv.get_dest_u8() as usize;
    history[side_index(side_to_move)][from][to]
}

pub fn order_moves(
    board: &Board,
    moves: &mut [BitMove],
    tt_move: Option<BitMove>,
    killers: [BitMove; 2],
    history: &HistoryTable,
    side_to_move: Player,
) {
    moves.sort_unstable_by(|a, b| {
        let sa = move_score(board, *a, tt_move, killers, history, side_to_move);
        let sb = move_score(board, *b, tt_move, killers, history, side_to_move);
        sb.cmp(&sa)
    });
}

pub fn order_captures(board: &Board, moves: &mut [BitMove]) {
    moves.sort_unstable_by(|a, b| mvv_lva_score(board, *b).cmp(&mvv_lva_score(board, *a)));
}
