use crate::utils::log_run_to_csv;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use crate::utils::print_placement_summary;

///---------------------orientation constant -------------------
const ORIENTS: [(usize, usize, usize); 6] = [
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
];

#[derive(Clone, Eq, Debug)]
pub struct ExtremePoint {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub bin_id: u32,
}
impl PartialEq for ExtremePoint {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z && self.bin_id == other.bin_id
    }
}

impl Hash for ExtremePoint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
        self.z.hash(state);
        self.bin_id.hash(state);
    }
}

/// Common packing metrics for both PPO and ACO solvers
#[derive(Debug, Serialize, Deserialize)]
pub struct PackingMetrics {
    pub placement_ratio: f64,
    pub volume_utilization: f64,
    pub bin_penalty: f64,
    pub mean_bin_waste: f64,
    pub score_eval: f64,
}

/// Extended metrics specific to ACO solver
#[derive(Debug, Serialize, Deserialize)]
pub struct AcoMetrics {
    #[serde(flatten)]
    pub base: PackingMetrics,
    pub iterations: usize,
    pub best_score: f64,
    pub total_time_sec: f64,
}

/// Wrapper for solver results with tagged enum
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "method")]
pub enum SolverResult {
    PPO {
        #[serde(flatten)]
        metrics: PackingMetrics,
    },
    ACO {
        #[serde(flatten)]
        metrics: AcoMetrics,
    },
}

#[derive(Deserialize)]
pub struct BinInput {
    pub id: u32,
    pub length: u32,
    pub width: u32,
    pub depth: u32,
}
#[derive(Deserialize)]
pub struct BoxInput {
    pub id: String,
    pub length: u32,
    pub width: u32,
    pub depth: u32,
}
#[derive(Deserialize)]
pub struct ProblemInput {
    pub bins: Vec<BinInput>,
    pub boxes: Vec<BoxInput>,
}
#[derive(Serialize)]
pub struct SolutionOutput {
    pub placements: Vec<Placement>,
    pub score: f64,
}

#[derive(Clone)]
pub struct Bin {
    pub id: u32,
    pub length: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Serialize)]
pub struct Placement {
    pub box_id: String,
    pub bin_id: u32,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Clone)]
pub struct BoxItem {
    pub id: String,
    pub length: u32,
    pub width: u32,
    pub height: u32,
}
impl BoxItem {
    /// Returns (L, W, H) after applying one of the 6 axis-aligned rotations.
    pub fn dims_for(&self, orient_idx: usize) -> (u32, u32, u32) {
        let (a, b, c) = ORIENTS[orient_idx];
        let dims = [self.length, self.width, self.height];
        (dims[a], dims[b], dims[c])
    }
}

pub struct Ant {
    pub path_taken: Vec<usize>,
    pub score: f64,
}

impl Ant {
    pub fn new(num_boxes: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut path: Vec<usize> = (0..num_boxes).collect();
        path.shuffle(&mut rng);
        Self {
            path_taken: path,
            score: 0.0,
        }
    }
}

pub struct AcoStruct {
    pub boxes: Vec<BoxItem>,
    pub bins: Vec<Bin>,
    pub pheromones: HashMap<(usize, ExtremePoint), f64>,
    pub pheromone_value: f64,
    pub decay: f64,
    pub number_of_iterations: usize,
    pub ant_count: usize,
    pub init_alpha: f64,
    pub init_beta: f64,
    pub best_order: Vec<usize>,
    pub best_score: f64,
    pub rank_limit: u32,

    // —— fields for stagnation control ——
    /// History of best scores at each iteration
    pub best_score_history: Vec<f64>,
    /// Number of consecutive no‐improvement iterations
    pub stagnation_counter: usize,
    /// Threshold of no‐improvement iterations to trigger reset
    pub stagnation_limit: usize,
    /// Minimum pheromone trail value
    pub pheromone_min: f64,
    /// Maximum pheromone trail value
    pub pheromone_max: f64,
}

impl AcoStruct {
    pub fn new(boxes: Vec<BoxItem>, bins: Vec<Bin>) -> Self {
        let mut pheromones = HashMap::new();
        for (i, _) in boxes.iter().enumerate() {
            for bin in &bins {
                let ep = ExtremePoint {
                    x: 0,
                    y: 0,
                    z: 0,
                    bin_id: bin.id,
                };
                pheromones.insert((i, ep), 1.0);
            }
        }
        let decay = 0.80;
        let number_of_iterations = 100;
        let stagnation_limit = (number_of_iterations as f64 * 0.1).ceil() as usize;
        let pheromone_max = 15.0;
        let pheromone_min = 2.0;

        Self {
            boxes,
            bins,
            pheromones,
            pheromone_value: 5.0,
            decay,
            number_of_iterations,
            ant_count: 100,
            init_alpha: 1.0,
            init_beta: 2.0,
            best_order: vec![],
            best_score: f64::MIN,
            rank_limit: 5,
            best_score_history: Vec::with_capacity(number_of_iterations),
            stagnation_counter: 0,
            stagnation_limit,
            pheromone_min,
            pheromone_max,
        }
    }
    pub fn new_dynamic(boxes: Vec<BoxItem>, bins: Vec<Bin>) -> Self {
        let b = boxes.len() as f64;
        let n = bins.len() as f64;

        let scale: f64 = 2.5; //for testing and inner program adjusting currently stable but could use some love
        let number_of_ants = (b * (n + 1.0).log2() * scale).round().clamp(50.0, 8000.0) as usize;
        let number_of_iterations = (n * (b + 1.0).log2() * (scale * 4.0))
            .round()
            .clamp(150.0, 5000.0) as usize;

        println!(
            "Using {} ants and {} iterations for {} boxes and {} bins",
            number_of_ants,
            number_of_iterations,
            boxes.len(),
            bins.len()
        );

        let mut model = AcoStruct::new(boxes, bins);
        model.ant_count = number_of_ants;
        model.number_of_iterations = number_of_iterations;
        // recompute stagnation limit based on dynamic run length
        model.stagnation_limit = (model.number_of_iterations as f64 * 0.05).ceil() as usize;
        model
    }

    pub fn run_model(&mut self) -> SolverResult {
        let start = Instant::now();
        self.run_batch();
        let total_time = start.elapsed().as_secs_f64();

        let (placements, score) = self.get_best_solution();
        // compute base metrics
        let placed = placements.len() as f64;
        let total_boxes = self.boxes.len() as f64;
        let placement_ratio = if total_boxes > 0.0 {
            placed / total_boxes
        } else {
            0.0
        };
        let used_vol: f64 = placements
            .iter()
            .map(|p| {
                let b = self.box_by_id(&p.box_id);
                (b.length * b.width * b.height) as f64
            })
            .sum();
        let bins_used: HashSet<u32> = placements.iter().map(|p| p.bin_id).collect();

        // total volume over opened bins
        let total_vol: f64 = self
            .bins
            .iter()
            .filter(|b| bins_used.contains(&b.id))
            .map(|b| (b.length * b.width * b.height) as f64)
            .sum::<f64>()
            .max(1.0);
        let volume_utilization = used_vol / total_vol;

        // normalized bin‐count penalty over opened bins
        let bin_penalty = bins_used.len() as f64 / (self.bins.len() as f64).max(1.0);

        // mean waste over only opened bins
        let mean_bin_waste: f64 = {
            let wastes: Vec<f64> = self
                .bins
                .iter()
                .filter(|b| bins_used.contains(&b.id))
                .map(|bin| {
                    let vol_b = (bin.length * bin.width * bin.height) as f64;
                    let used_b: f64 = placements
                        .iter()
                        .filter(|p| p.bin_id == bin.id)
                        .map(|p| {
                            let b = self.box_by_id(&p.box_id);
                            (b.length * b.width * b.height) as f64
                        })
                        .sum();
                    1.0 - used_b / vol_b
                })
                .collect();
            if wastes.is_empty() {
                0.0
            } else {
                wastes.iter().sum::<f64>() / (wastes.len() as f64)
            }
        };

        let base = PackingMetrics {
            placement_ratio,
            volume_utilization,
            bin_penalty,
            mean_bin_waste,
            score_eval: score,
        };
        let metrics = AcoMetrics {
            base,
            iterations: self.number_of_iterations,
            best_score: self.best_score,
            total_time_sec: total_time,
        };
        SolverResult::ACO { metrics }
    }

    pub fn run_batch(&mut self) {
        let mut best_iteration = 0;
        let mut _all_ants: Vec<Ant> = Vec::new();
        let mut _all_paths: Vec<Vec<(usize, ExtremePoint)>> = Vec::new();

        for iteration in 0..self.number_of_iterations {
            let mut iteration_ants = Vec::new();
            let mut iteration_paths = Vec::new();
            let alpha = self.init_alpha
                * (1.0 - iteration as f64 / self.number_of_iterations as f64)
                + 10.0 * (iteration as f64 / self.number_of_iterations as f64);
            let beta = self.init_beta * (1.0 - iteration as f64 / self.number_of_iterations as f64)
                + 1.0 * (iteration as f64 / self.number_of_iterations as f64);

            for _ in 0..self.ant_count {
                let mut ant = Ant::new(self.boxes.len());
                let (score, _, placement_path) = self.evaluate_order(&ant.path_taken, alpha, beta);
                ant.score = score;
                iteration_ants.push(ant);
                iteration_paths.push(placement_path);
            }

            iteration_ants.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

            if iteration_ants[0].score > self.best_score {
                self.best_order = iteration_ants[0].path_taken.clone();
                self.best_score = iteration_ants[0].score;
                best_iteration = iteration;
            }

            println!(
                "Iteration {} best score: {:.4}",
                iteration, iteration_ants[0].score
            );
            self.update_pheromones_batch(&iteration_ants, &iteration_paths);

            // —— stagnation control start ——
            self.best_score_history.push(self.best_score);
            if self.best_score_history.len() > 1 {
                let prev = self.best_score_history[self.best_score_history.len() - 2];
                if self.best_score <= prev {
                    self.stagnation_counter += 1;
                } else {
                    self.stagnation_counter = 0;
                }
            }
            if self.stagnation_counter >= self.stagnation_limit {
                self.soft_reset_pheromones();
                self.stagnation_counter = 0;
            }
            // clamp all trails to [min, max]
            for v in self.pheromones.values_mut() {
                *v = v.clamp(self.pheromone_min, self.pheromone_max);
            }
            // —— stagnation control end ——

            _all_ants = iteration_ants;
            _all_paths = iteration_paths;
        }

        println!(
            "Best score: {:.4} found at iteration {}",
            self.best_score, best_iteration
        );
        let (placements, _) = self.get_best_solution();

        let used_bins: std::collections::HashSet<u32> =
            placements.iter().map(|p| p.bin_id).collect();

        log_run_to_csv(
            "aco_run_log.csv",
            self.boxes.len(),
            self.bins.len(),
            self.ant_count,
            self.number_of_iterations,
            self.best_score,
            placements.len(),
            used_bins.len(),
            best_iteration,
        );
    }
    pub fn evaluate_order(
        &mut self,
        order: &[usize],
        alpha: f64,
        beta: f64,
    ) -> (f64, Vec<Placement>, Vec<(usize, ExtremePoint)>) {
        let mut placement_path = Vec::new();
        let mut placements = Vec::new();
        let mut occupied: HashMap<u32, Vec<Placement>> = HashMap::new();
        let mut ep_list: Vec<ExtremePoint> = self
            .bins
            .iter()
            .map(|bin| ExtremePoint {
                x: 0,
                y: 0,
                z: 0,
                bin_id: bin.id,
            })
            .collect();

        for &box_idx in order {
            let b = &self.boxes[box_idx];
            let mut feasible: Vec<(ExtremePoint, usize, Placement)> = vec![];

            for ep in &ep_list {
                let bin = self.bin_by_id(ep.bin_id);
                for orient in 0..ORIENTS.len() {
                    let (L, W, H) = b.dims_for(orient);
                    if ep.x + L > bin.length || ep.y + W > bin.width || ep.z + H > bin.height {
                        continue;
                    }
                    let placement = Placement {
                        box_id: b.id.clone(),
                        bin_id: ep.bin_id,
                        x: ep.x,
                        y: ep.y,
                        z: ep.z,
                    };
                    let binding = Vec::new();
                    let existing = occupied.get(&ep.bin_id).unwrap_or(&binding);
                    if !self.overlaps_with_dims(&placement, existing, L, W, H) {
                        feasible.push((ep.clone(), orient, placement));
                    }
                }
            }

            if feasible.is_empty() {
                continue;
            }

            let weights: Vec<f64> = feasible
                .iter()
                .map(|(ep, orient, _)| {
                    let ep_q = self.quantize_ep(ep, 50);
                    let tau = *self.pheromones.get(&(box_idx, ep_q)).unwrap_or(&1.0);
                    // gravity heuristic to simulate gravity
                    let eta = 1.0 / (ep.z as f64 + 1.0);
                    tau.powf(alpha) * eta.powf(beta)
                })
                .collect();
            let dist = WeightedIndex::new(&weights).unwrap();
            let idx = dist.sample(&mut thread_rng());
            let (chosen_ep, chosen_orient, placement) = feasible.swap_remove(idx);

            placement_path.push((box_idx, chosen_ep.clone()));
            placements.push(placement.clone());
            occupied
                .entry(chosen_ep.bin_id)
                .or_default()
                .push(placement.clone());

            let (L, W, H) = b.dims_for(chosen_orient);
            ep_list.sort_by_key(|ep| ep.z);
            ep_list.truncate(1000);
            self.update_extreme_points_with_dims(
                &mut ep_list,
                &placement,
                L,
                W,
                H,
                self.bin_by_id(chosen_ep.bin_id),
            );
        }

        // metrics exactly the same as in PPO
        let placed = placements.len() as f64;
        let total = self.boxes.len() as f64;
        let pr = if total > 0.0 { placed / total } else { 0.0 };

        let used_vol: f64 = placements
            .iter()
            .map(|p| {
                let b = self.box_by_id(&p.box_id);
                let (l, w, h) = (b.length, b.width, b.height);
                (l * w * h) as f64
            })
            .sum();
        // only opened bins count
        let bins_used: HashSet<u32> = placements.iter().map(|p| p.bin_id).collect();

        // utilization over opened bins’ total volume
        let total_vol: f64 = self
            .bins
            .iter()
            .filter(|b| bins_used.contains(&b.id))
            .map(|b| (b.length * b.width * b.height) as f64)
            .sum::<f64>()
            .max(1.0);
        let vu = used_vol / total_vol;

        // same normalized penalty as above
        let bp = bins_used.len() as f64 / (self.bins.len() as f64).max(1.0);

        // mean waste over only the opened bins
        let mu_w = {
            let wastes: Vec<f64> = self
                .bins
                .iter()
                .filter(|b| bins_used.contains(&b.id))
                .map(|bin| {
                    let vol_b = (bin.length * bin.width * bin.height) as f64;
                    let used_b: f64 = placements
                        .iter()
                        .filter(|p| p.bin_id == bin.id)
                        .map(|p| {
                            let b = self.box_by_id(&p.box_id);
                            (b.length * b.width * b.height) as f64
                        })
                        .sum();
                    1.0 - used_b / vol_b
                })
                .collect();
            if wastes.is_empty() {
                0.0
            } else {
                wastes.iter().sum::<f64>() / (wastes.len() as f64)
            }
        };

        // PPO based weights scoring
        const GAMMA: f64 = 0.65;
        const DELTA: f64 = 0.25;
        const EPSILON: f64 = 0.30;
        const ZETA: f64 = 0.25;
        let score = (GAMMA * pr + DELTA * vu - EPSILON * bp - ZETA * mu_w).clamp(0.0, 1.0);

        (score, placements, placement_path)
    }

    // Helper overlap check that uses explicit dims rather than looking up BoxItem dimensions:
    fn overlaps_with_dims(
        &self,
        new: &Placement,
        existing: &[Placement],
        L: u32,
        W: u32,
        H: u32,
    ) -> bool {
        for p in existing.iter().filter(|p| p.bin_id == new.bin_id) {
            let (l2, w2, h2) = {
                let b = self.box_by_id(&p.box_id);
                (b.length, b.width, b.height)
            };
            let a_min = (new.x, new.y, new.z);
            let a_max = (new.x + L, new.y + W, new.z + H);
            let b_min = (p.x, p.y, p.z);
            let b_max = (p.x + l2, p.y + w2, p.z + h2);
            let overlap = a_min.0 < b_max.0
                && a_max.0 > b_min.0
                && a_min.1 < b_max.1
                && a_max.1 > b_min.1
                && a_min.2 < b_max.2
                && a_max.2 > b_min.2;
            if overlap {
                return true;
            }
        }
        false
    }

    // And a variant of update_extreme_points that takes explicit dims:
    pub fn update_extreme_points_with_dims(
        &self,
        ep_list: &mut Vec<ExtremePoint>,
        placed: &Placement,
        L: u32,
        W: u32,
        H: u32,
        bin: &Bin,
    ) {
        let mut new_eps = vec![
            ExtremePoint {
                x: placed.x + L,
                y: placed.y,
                z: placed.z,
                bin_id: bin.id,
            },
            ExtremePoint {
                x: placed.x,
                y: placed.y + W,
                z: placed.z,
                bin_id: bin.id,
            },
            ExtremePoint {
                x: placed.x,
                y: placed.y,
                z: placed.z + H,
                bin_id: bin.id,
            },
        ];
        new_eps.retain(|ep| ep.x <= bin.length && ep.y <= bin.width && ep.z <= bin.height);
        ep_list.extend(new_eps);
    }

    fn box_by_id(&self, id: &str) -> &BoxItem {
        self.boxes
            .iter()
            .find(|b| b.id == id)
            .expect("Box ID not found")
    }
    pub fn bin_by_id(&self, id: u32) -> &Bin {
        self.bins
            .iter()
            .find(|b| b.id == id)
            .expect("Bin ID not found")
    }
    pub fn soft_reset_pheromones(&mut self) {
        for val in self.pheromones.values_mut() {
            *val = self.pheromone_value / 2.0;
        }
    }

    pub fn update_pheromones_batch(
        &mut self,
        ants: &[Ant],
        all_placements: &[Vec<(usize, ExtremePoint)>],
    ) {
        // 1. Evaporate
        for val in self.pheromones.values_mut() {
            *val *= self.decay;
        }

        // 2. Reinforce pheromones based on rank and score
        for (rank, (ant, placement_path)) in ants
            .iter()
            .zip(all_placements)
            .take(self.rank_limit as usize)
            .enumerate()
        {
            let weight = (self.rank_limit - rank as u32) as f64 / self.rank_limit as f64;

            for (box_idx, ep) in placement_path {
                let ep_q = self.quantize_ep(ep, 50);
                let entry = self.pheromones.entry((*box_idx, ep_q)).or_insert(1.0);
                *entry += self.pheromone_value * weight * ant.score;
            }
        }
    }

    pub fn get_best_solution(&self) -> (Vec<Placement>, f64) {
        use std::collections::HashMap;

        let mut placements = Vec::new();
        let mut occupied: HashMap<u32, Vec<Placement>> = HashMap::new();
        let mut ep_list: Vec<ExtremePoint> = self
            .bins
            .iter()
            .map(|bin| ExtremePoint {
                x: 0,
                y: 0,
                z: 0,
                bin_id: bin.id,
            })
            .collect();

        for &box_idx in &self.best_order {
            let b = &self.boxes[box_idx];

            let feasible: Vec<(ExtremePoint, Placement)> = ep_list
                .iter()
                .filter_map(|ep| {
                    let bin = self.bin_by_id(ep.bin_id);
                    if ep.x + b.length > bin.length
                        || ep.y + b.width > bin.width
                        || ep.z + b.height > bin.height
                    {
                        return None;
                    }

                    let placement = Placement {
                        box_id: b.id.clone(),
                        bin_id: ep.bin_id,
                        x: ep.x,
                        y: ep.y,
                        z: ep.z,
                    };
                    let binding = Vec::new();
                    let existing = occupied.get(&ep.bin_id).unwrap_or(&binding);
                    if !self.overlaps_with_dims(&placement, existing, b.length, b.width, b.height) {
                        Some((ep.clone(), placement))
                    } else {
                        None
                    }
                })
                .collect();

            if let Some((best_ep, placement)) = feasible
                .iter()
                .max_by(|(ep1, _), (ep2, _)| {
                    let p1 = *self
                        .pheromones
                        .get(&(box_idx, self.quantize_ep(ep1, 50)))
                        .unwrap_or(&1.0);
                    let p2 = *self
                        .pheromones
                        .get(&(box_idx, self.quantize_ep(ep2, 50)))
                        .unwrap_or(&1.0);
                    p1.partial_cmp(&p2).unwrap()
                })
                .cloned()
            {
                placements.push(placement.clone());
                occupied
                    .entry(best_ep.bin_id)
                    .or_default()
                    .push(placement.clone());
                // update EP with explicit dims
                ep_list.sort_by_key(|ep| ep.z);
                //cut eps to reduce compute
                ep_list.truncate(1000);
                self.update_extreme_points_with_dims(
                    &mut ep_list,
                    &placement,
                    b.length,
                    b.width,
                    b.height,
                    self.bin_by_id(best_ep.bin_id),
                );
            }
        }

        let placed_boxes = placements.len();
        let total_boxes = self.boxes.len();
        let placement_ratio = (placements.len()/self.boxes.len()) as f64;

        let bins_used: HashSet<u32> = placements.iter().map(|p| p.bin_id).collect();
        let bin_count = bins_used.len();

        let used_volume: usize = placements
            .iter()
            .map(|p| {
                let b = self.box_by_id(&p.box_id);
                (b.length * b.width * b.height) as usize
            })
            .sum();

        // only opened bins count
        let bins_used: HashSet<u32> = placements.iter().map(|p| p.bin_id).collect();

        // total volume over opened bins
        let total_bin_volume: usize = self
            .bins
            .iter()
            .filter(|b| bins_used.contains(&b.id))
            .map(|b| (b.length * b.width * b.height) as usize)
            .sum::<usize>()
            .max(1);
        let volume_utilization = used_volume as f64 / (total_bin_volume as f64);

        // normalized penalty
        let bin_penalty = bins_used.len() as f64 / (self.bins.len() as f64).max(1.0);

        // mean waste over only opened bins
        let mu_w = {
            let wastes: Vec<f64> = self
                .bins
                .iter()
                .filter(|b| bins_used.contains(&b.id))
                .map(|bin| {
                    let vol_b = (bin.length * bin.width * bin.height) as f64;
                    let used_b: f64 = placements
                        .iter()
                        .filter(|p| p.bin_id == bin.id)
                        .map(|p| {
                            let b = self.box_by_id(&p.box_id);
                            (b.length * b.width * b.height) as f64
                        })
                        .sum();
                    1.0 - used_b / vol_b
                })
                .collect();
            if wastes.is_empty() {
                0.0
            } else {
                wastes.iter().sum::<f64>() / (wastes.len() as f64)
            }
        };

        let score =
            0.65 * placement_ratio + 0.25 * volume_utilization - 0.3 * bin_penalty - 0.25 * mu_w;

        (placements, score.max(0.0))
    }
    //unused function meant for further research in the online problem
    pub fn _place_next_box(
        &mut self,
        box_item: &BoxItem,
        bin_states: &[Vec<Placement>],
    ) -> Placement {
        let bin_id = self.bins[0].id;
        Placement {
            box_id: box_item.id.clone(),
            bin_id,
            x: 0,
            y: 0,
            z: 0,
        }
    }
    fn quantize_ep(&self, ep: &ExtremePoint, bucket: u32) -> ExtremePoint {
        ExtremePoint {
            x: (ep.x / bucket) * bucket,
            y: (ep.y / bucket) * bucket,
            z: (ep.z / bucket) * bucket,
            bin_id: ep.bin_id,
        }
    }
}

//data operations taken out of the main clump of code
impl AcoStruct {
    //data communication
    pub fn from_json(path: &str) -> Self {
        let data = fs::read_to_string(path).expect("Failed to read input file");
        let parsed: ProblemInput = serde_json::from_str(&data).expect("Invalid JSON format");

        let boxes: Vec<BoxItem> = parsed
            .boxes
            .into_iter()
            .map(|b| BoxItem {
                id: b.id.to_string(), // if BoxInput uses u32, convert to string
                length: b.length,
                width: b.width,
                height: b.depth, // depth = height
            })
            .collect();

        let bins: Vec<Bin> = parsed
            .bins
            .into_iter()
            .map(|b| Bin {
                id: b.id,
                length: b.length,
                width: b.width,
                height: b.depth, // same assumption
            })
            .collect();

        AcoStruct::new_dynamic(boxes, bins)
    }

    pub fn to_json(&self, path: &str) {
        let (placements, score) = self.get_best_solution();
        print_placement_summary(&placements);
        let out = SolutionOutput { placements, score };
        let json = serde_json::to_string_pretty(&out).expect("Failed to serialize output");
        fs::write(path, json).expect("Failed to write output file");
    }
}

//good looking metrics for early testing
impl AcoStruct {
    pub fn get_params(&self) {
        println!(
            "Pheromones: {:?} \n
                  Decay: {:?} \n 
                  number_of_iterations: {:?} \n
                  ant_count: {:?} \n
                  init_alpha: {:?} \n
                  init beta: {:?}  \n
                  rank_limit: {:?} \n",
            self.pheromones,
            self.decay,
            self.number_of_iterations,
            self.ant_count,
            self.init_alpha,
            self.init_beta,
            self.rank_limit
        )
    }
}

//primitive early test
#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::{Bin, BoxItem};

    #[test]
    fn test_batch_run() {
        let boxes = vec![
            BoxItem {
                id: "A".into(),
                length: 100,
                width: 100,
                height: 100,
            },
            BoxItem {
                id: "B".into(),
                length: 100,
                width: 100,
                height: 100,
            },
        ];
        let bins = vec![Bin {
            id: 1,
            length: 500,
            width: 500,
            height: 500,
        }];
        let mut model = AcoStruct::new_dynamic(boxes, bins);
        model.run_model();
        let (placements, score) = model.get_best_solution();
        assert!(!placements.is_empty());
        assert!(score > 0.0);
    }
}
