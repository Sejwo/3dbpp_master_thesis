import os
import json
import argparse
import tempfile
import csv
import torch
import time
import aco3dbpp
from train_ppo import OfflinePackingEnv, SimplePPOPolicy, gen_random_instances
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ORIENTS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def generate_instances_json(path, num_instances=1):
    instances = gen_random_instances(num_instances)
    data = instances[0] if num_instances == 1 else {'instances': instances}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Generated {num_instances} instance(s) to {path}")


def verify_in_bounds(inst, res):
    ##Sanity‐check that all placements are within the bin dimensions.
    bins = inst["bins"]
    boxes = {b["id"]: b for b in inst["boxes"]}
    errors = []
    for p in res.get("placements", []):
        bid, x, y, z = p["bin_id"], p["x"], p["y"], p["z"]
        b_spec = bins[bid]
        dx = boxes[p["box_id"]].get("length", boxes[p["box_id"]].get("x"))
        dy = boxes[p["box_id"]].get("width",  boxes[p["box_id"]].get("y"))
        dz = boxes[p["box_id"]].get("depth",  boxes[p["box_id"]].get("z"))
        if x + dx > b_spec["length"] or y + dy > b_spec["width"] or z + dz > b_spec["depth"]:
            errors.append((p, b_spec))
    if errors:
        print("Found OUT-OF-BOUNDS placements:")
        for p, b in errors:
            print("   ", p, "vs bin dims", b)
    else:
        print("All placements for score", res.get("score", res.get("score_eval")), "are within bounds.")


def load_instances(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and 'instances' in data:
        return data['instances']
    if isinstance(data, dict) and 'bins' in data and 'boxes' in data:
        return [data]
    raise ValueError(f"Unexpected JSON format in {path}")


def run_ppo_instance(inst, policy):
    """
    Runs one instance through the PPO policy and returns a dict with:
      - placements (list of {box_id, bin_id, x, y, z})
      - elapsed time
      - placement_ratio, volume_utilization, bin_penalty, mean_bin_waste, score_eval
    """
    bins = inst['bins']
    boxes = inst['boxes']
    start = time.perf_counter()

    env = OfflinePackingEnv(bins, boxes)
    obs = env.reset()
    done = False
    placements = []

    # 1) Let PPO place boxes until done
    while not done:
        specs = torch.tensor(obs['box_specs']).unsqueeze(0)
        mask = torch.tensor(obs['placed_mask']).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(specs, mask)
        action = torch.argmax(logits, dim=1).item()
        obs, _, done, info = env.step(action)
        if 'placement' in info:
            placements.append(info['placement'])

    # 2) Reconstruct the placements list in the same format ACO uses
    placements = [
        {
            'box_id': boxes[idx]['id'],
            'bin_id': bin_id,
            'x': x, 'y': y, 'z': z
        }
        for (idx, x, y, z, bin_id) in env.positions
    ]
    elapsed = time.perf_counter() - start

    # 3) Compute placement_ratio
    total_boxes = len(boxes)
    pr = len(placements) / total_boxes if total_boxes else 0.0
    # 4) Compute used_volume by summing each placed box’s actual volume
    used_volume = 0.0
    for p in placements:
        b_spec = next(b for b in boxes if b['id'] == p['box_id'])
        dx = b_spec.get('length', b_spec.get('x'))
        dy = b_spec.get('width',  b_spec.get('y'))
        dz = b_spec.get('depth',  b_spec.get('z'))
        used_volume += dx * dy * dz

    # 5) Compute set of opened bins
    used_bins = {p['bin_id'] for p in placements}

    # 6) Volume utilization: sum volume of only those bins that were opened
    total_bin_volume = 0.0
    for b_spec in bins:
        if b_spec['id'] in used_bins:
            bx = b_spec.get('length', b_spec.get('x'))
            by = b_spec.get('width',  b_spec.get('y'))
            bz = b_spec.get('depth',  b_spec.get('z'))
            total_bin_volume += (bx * by * bz)
    if total_bin_volume == 0.0:
        total_bin_volume = 1.0
    vu = used_volume / total_bin_volume

    # 7) Bin penalty: fraction of bins used
    bp = len(used_bins) / len(bins) if bins else 0.0

    # 8) Mean bin waste: iterate only over opened bins
    wastes = []
    for b_spec in bins:
        if b_spec['id'] not in used_bins:
            continue
        bx = b_spec.get('length', b_spec.get('x'))
        by = b_spec.get('width',  b_spec.get('y'))
        bz = b_spec.get('depth',  b_spec.get('z'))
        vol_b = (bx * by * bz) or 1.0

        # sum volumes of boxes inside this particular bin
        used_b = 0.0
        for p in placements:
            if p['bin_id'] != b_spec['id']:
                continue
            inner = next(b for b in boxes if b['id'] == p['box_id'])
            dx = inner.get('length', inner.get('x'))
            dy = inner.get('width',  inner.get('y'))
            dz = inner.get('depth',  inner.get('z'))
            used_b += (dx * dy * dz)

        wastes.append(1.0 - used_b / vol_b)

    mu_w = (sum(wastes) / len(wastes)) if wastes else 0.0

    # 9) Final normalized score
    score_eval = max(0.65 * pr + 0.25 * vu - 0.30 * bp - 0.25 * mu_w, 0.0)

    return {
        'placements': placements,
        'time': elapsed,
        'placement_ratio': pr,
        'volume_utilization': vu,
        'bin_penalty': bp,
        'mean_bin_waste': mu_w,
        'score_eval': score_eval,
    }


def run_aco_instance(inst):
    """
    Writes the instance to a temp file, calls into the ACO binary, reads back placements + metrics.
    Returns a dict containing placements + all ACO metrics.
    """
    tmp_in = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    tmp_out = tempfile.NamedTemporaryFile(mode='r', suffix='.json', delete=False)
    try:
        json.dump(inst, tmp_in)
        tmp_in.flush()
        tmp_in.close()
        tmp_out.close()

        # Call ACO to generate placements + raw score
        aco3dbpp.solve_and_write(tmp_in.name, tmp_out.name)
        with open(tmp_out.name, 'r') as f:
            out_data = json.load(f)
        placements = out_data.get('placements', [])
        raw_score = out_data.get('score')

        # Get all flattened metrics from ACO
        metrics_json = aco3dbpp.solve_from_json(tmp_in.name)
        metrics_data = json.loads(metrics_json)
        # drop the "method" tag
        metrics_data.pop('method', None)

        return {
            'placements': placements,
            'placement_ratio':    metrics_data['placement_ratio'],
            'volume_utilization': metrics_data['volume_utilization'],
            'bin_penalty':        metrics_data['bin_penalty'],
            'mean_bin_waste':     metrics_data['mean_bin_waste'],
            'score_eval':         metrics_data['score_eval'],
            'iterations':         metrics_data['iterations'],
            'best_score':         metrics_data.get('best_cost', metrics_data.get('best_score')),
            'total_time_sec':     metrics_data['total_time_sec'],
            'solver_raw_score':   raw_score,
        }
    finally:
        os.remove(tmp_in.name)
        os.remove(tmp_out.name)


def visualise_solutions(inst, ppo_res, aco_res):
    """
    Draws a 3D bar plot for each bin in PPO and ACO placements side by side.
    """
    n_bins = len(inst['bins'])
    fig = plt.figure(figsize=(5 * n_bins, 10))

    for alg_i, (label, res) in enumerate([('PPO', ppo_res), ('ACO', aco_res)]):
        for bin_i, bin_spec in enumerate(inst['bins']):
            ax = fig.add_subplot(
                2, n_bins,
                alg_i * n_bins + bin_i + 1,
                projection='3d'
            )

            # Get bin dimensions
            if 'dimensions' in bin_spec:
                bx, by, bz = bin_spec['dimensions']
            else:
                bx = bin_spec.get('length', bin_spec.get('x'))
                by = bin_spec.get('width',  bin_spec.get('y'))
                bz = bin_spec.get('depth',  bin_spec.get('z'))

            # Draw the empty bin as a transparent wireframe
            corners = [
                (0, 0, 0), (bx, 0, 0), (bx, by, 0), (0, by, 0),
                (0, 0, bz), (bx, 0, bz), (bx, by, bz), (0, by, bz)
            ]
            faces = [
                [0,1,2,3], [4,5,6,7], [0,1,5,4],
                [2,3,7,6], [1,2,6,5], [0,3,7,4]
            ]
            face_verts = [[corners[idx] for idx in quad] for quad in faces]
            bin_col = Poly3DCollection(face_verts, edgecolors='black', linewidths=1)
            bin_col.set_facecolor((0,0,0,0))  # fully transparent
            ax.add_collection3d(bin_col)

            # Plot all boxes assigned to this bin
            box_map = {b['id']: b for b in inst['boxes']}
            for placement in res.get('placements', []):
                if placement['bin_id'] != bin_i:
                    continue
                px, py, pz = placement['x'], placement['y'], placement['z']
                bid = placement['box_id']
                box = box_map[bid]
                l = box.get('length', box.get('x'))
                w = box.get('width',   box.get('y'))
                d = box.get('depth',   box.get('z'))
                # Find an orientation that fits entirely within the bin
                for a, b, c in ORIENTS:
                    dx, dy, dz = [l, w, d][a], [l, w, d][b], [l, w, d][c]
                    if px + dx > bx or py + dy > by or pz + dz > bz:
                        continue
                    _plot_cuboid(ax, px, py, pz, dx, dy, dz)
                    break

            ax.set_title(f"{label} – bin {bin_i}")
            ax.set_xlim(0, bx)
            ax.set_ylim(0, by)
            ax.set_zlim(0, bz)
            ax.set_box_aspect((bx, by, bz))

    plt.tight_layout()
    plt.show()


def _plot_cuboid(ax, x, y, z, dx, dy, dz):
    """
    Draw one cuboid as a 3D bar to avoid Poly3DCollection projection errors.
    """
    ax.bar3d(
        x, y, z,      # origin
        dx, dy, dz,   # lengths along each axis
        shade=True,
        alpha=0.3,
        color='cyan',
        edgecolor='black',
        linewidth=1
    )


def visualise_random(policy):
    """
    Generate a single random instance, solve via PPO and ACO, and visualize.
    """
    inst = gen_random_instances(1)[0]
    inst_dict = {'bins': inst['bins'], 'boxes': inst['boxes']}
    ppo_res = run_ppo_instance(inst_dict, policy)
    with open('ppo_solution.json', 'w') as f:
        json.dump(ppo_res, f, indent=2)
    print('Wrote PPO solution to ppo_solution.json')

    aco_res = run_aco_instance(inst_dict)
    with open('aco_solution.json', 'w') as f:
        json.dump(aco_res, f, indent=2)
    print('Wrote ACO solution to aco_solution.json')

    verify_in_bounds(inst_dict, ppo_res)
    verify_in_bounds(inst_dict, aco_res)
    visualise_solutions(inst_dict, ppo_res, aco_res)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark PPO vs ACO with optional visualization'
    )
    parser.add_argument('--generate-json', help='Write random instances to JSON')
    parser.add_argument('--num-instances', type=int, default=1)
    parser.add_argument('--json-input', help='Path to instances JSON')
    parser.add_argument('--output-csv', help='Path for CSV results')
    parser.add_argument('--ppo-model-path', default='ppo_offline_bpp_20250522_1010.pt')
    parser.add_argument('--visualise', action='store_true', help='Visualise a new random instance')
    args = parser.parse_args()

    if args.generate_json:
        generate_instances_json(args.generate_json, args.num_instances)
        return

    # Load PPO policy
    policy = SimplePPOPolicy()
    if not os.path.isfile(args.ppo_model_path):
        parser.error(f'PPO checkpoint not found: {args.ppo_model_path}')
    state = torch.load(args.ppo_model_path, map_location='cpu')
    policy.load_state_dict(state)
    policy.eval()

    if args.visualise:
        visualise_random(policy)
        return

    if not args.json_input or not os.path.isfile(args.json_input):
        parser.error('--json-input is required and must exist')

    instances = load_instances(args.json_input)
    # Run benchmark and collect rows
    rows = []
    for idx, inst in enumerate(instances):
        ppo = run_ppo_instance(inst, policy)
        aco = run_aco_instance(inst)

        combined = {'instance_index': idx}
        for k, v in ppo.items():
            if k == 'placements':
                continue
            combined[f'ppo-{k}'] = v
        for k, v in aco.items():
            if k == 'placements':
                continue
            combined[f'aco-{k}'] = v

        rows.append(combined)
        print(
            f"Instance {idx}: "
            f"PPO score={ppo['score_eval']:.4f}, "
            f"ACO score={aco.get('best_score', aco.get('score', float('nan'))):.4f}"
        )

    fieldnames = sorted(rows[0].keys()) if rows else []
    if args.output_csv:
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {args.output_csv}")
    else:
        print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
