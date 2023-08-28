use std::cmp::Ordering;

use pleco::{Board, Piece, BitMove, PieceType};
use tch::Tensor;
use ndarray::Array2;

fn piece_value(piece_type: PieceType) -> i32 {
    match piece_type {
        PieceType::P => 1,
        PieceType::N => 3,
        PieceType::B => 3,
        PieceType::R => 5,
        PieceType::Q => 9,
        PieceType::K => 1000,
        _ => 0,
    }
}

// MVV-LVA heuristic
fn mvv_lva_score(move_: BitMove, board: &pleco::Board) -> i32 {
    let destination_square = move_.get_dest();
    let attacker_square = move_.get_src();

    let victim_value = piece_value(board.piece_at_sq(destination_square).type_of());
    let attacker_value = piece_value(board.piece_at_sq(attacker_square).type_of());

    // The formula for MVV-LVA, prioritizing moves with valuable victims and low-value attackers
    1000 * victim_value - attacker_value
}


pub fn order_moves(board: &pleco::Board, killer_move: Option<BitMove>, best_move: Option<BitMove>) -> Vec<BitMove> {
    let mut moves = board.generate_moves().to_vec();

    // Sort the moves by MVV-LVA score
    moves.sort_by(|&a, &b| mvv_lva_score(b, board).cmp(&mvv_lva_score(a, board)));

    // Prioritize best_move and killer_move
    moves.sort_by(|&a, &b| {
        if Some(a) == best_move {
            Ordering::Less
        } else if Some(b) == best_move {
            Ordering::Greater
        } else if Some(a) == killer_move {
            Ordering::Less
        } else if Some(b) == killer_move {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });

    moves
}


pub fn board_to_bitboard(board: &Board) -> Tensor {
    let mut bitboard: Array2<f32> = Array2::<f32>::zeros([1, 775]);

    for (sq, piece) in board.get_piece_locations() {
        let offset = piece_to_offset(piece) + (sq.file_idx_of_sq() + sq.rank_idx_of_sq()*8) as i32;
        bitboard[[0, offset as usize]] = 1.0;
    }

    let turn = board.turn();

    match turn {
        pleco::Player::White => bitboard[[0, 768]] = 1.0,
        pleco::Player::Black => bitboard[[0, 768]] = 0.0,
    }

    if board.in_check() {
        if turn == pleco::Player::White {
            bitboard[[0, 769]] = 1.0;
        } else {    
            bitboard[[0, 770]] = 1.0;
        }
    }

    bitboard[[0, 771]] = board.can_castle(pleco::Player::White, pleco::core::CastleType::KingSide).into();
    bitboard[[0, 772]] = board.can_castle(pleco::Player::White, pleco::core::CastleType::QueenSide).into();
    bitboard[[0, 773]] = board.can_castle(pleco::Player::Black, pleco::core::CastleType::KingSide).into();
    bitboard[[0, 774]] = board.can_castle(pleco::Player::Black, pleco::core::CastleType::QueenSide).into();

    Tensor::from(bitboard.as_slice().unwrap())
}

pub fn piece_to_offset(piece: Piece) -> i32 {
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

        _ => panic!("Invalid piece"),
    }
}


