use pleco::Board;

mod search_algorithm;
mod utils;

fn main() {
    let mut b = Board::default();

    let s = search_algorithm::SearchAlgorithm::new(
        tch::CModule::load("models/eval_params264k_norm_mse0.117666_jit.pt").unwrap().into()
    );

    let m = s.search(&mut b, 4, 100000000.0);
    println!("{}", m);
}
