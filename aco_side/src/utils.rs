use std::collections::HashMap;
use crate::structs::{AcoStruct, Bin,Placement, ProblemInput, BoxItem, SolutionOutput};
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Utc;

//debugging function, removed from ready version
pub fn print_placement_summary(placements: &[Placement]) {
    let mut bins: HashMap<u32, Vec<&Placement>> = HashMap::new();
    for p in placements {
        bins.entry(p.bin_id).or_default().push(p);
    }

    for (bin_id, ps) in &bins {
        println!("\nCrude debug: Bin {bin_id} contains:");
        for p in ps {
            println!(
                "  • Box {} at ({}, {}, {})",
                p.box_id, p.x, p.y, p.z
            );
        }
    }
}
//logging for testing, can reuse sometime
pub fn log_run_to_csv(
    path: &str,
    boxes: usize,
    bins: usize,
    ants: usize,
    iterations: usize,
    best_score: f64,
    placed_boxes: usize,
    used_bins: usize,
    convergence_iter: usize,
) {
    let timestamp = Utc::now().to_rfc3339();

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("Cannot open CSV file");

    let header = "timestamp,boxes,bins,ants,iterations,best_score,placed_boxes,used_bins,convergence_iter\n";
    if file.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
        let _ = file.write_all(header.as_bytes());
    }

    let record = format!(
        "{},{},{},{},{},{:.4},{},{},{}\n",
        timestamp, boxes, bins, ants, iterations, best_score, placed_boxes, used_bins, convergence_iter
    );

    let _ = file.write_all(record.as_bytes());
}
//early tests before the structure of the code was finished
pub fn solve_offline_bpp(problem: &ProblemInput) -> SolutionOutput {
    let boxes: Vec<BoxItem> = problem
        .boxes
        .iter()
        .map(|b| BoxItem {
            id: b.id.clone(),
            length: b.length,
            width: b.width,
            height: b.depth, // depth == height
        })
        .collect();
    let bins: Vec<Bin> = problem
        .bins
        .iter()
        .map(|b| Bin {
            id: b.id,
            length: b.length,
            width: b.width,
            height: b.depth,
        })
        .collect();

    let mut model = AcoStruct::new_dynamic(boxes, bins);
    model.run_model();

    let (placements, score) = model.get_best_solution();
    SolutionOutput { placements, score }
}