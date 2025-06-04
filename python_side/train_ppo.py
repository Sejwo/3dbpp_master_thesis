import os
import sys
import json
import time
import random
import logging
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from py3dbp import Packer, Bin, Item

# Use all available CPU threads for PyTorch
torch.set_num_threads(os.cpu_count())
torch.set_num_interop_threads(os.cpu_count())

# Config
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
SINGLE_JSON_DIR = Path("input")
MODEL_OUTPUT = Path(f"ppo_offline_bpp_{timestamp}.pt")
EPISODES = 2000
GAMMA = 0.99
COMP_LOG_CSV = Path(f"comp_logs_{timestamp}.csv")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("OfflinePPO")

# Load instance from JSON
def load_instance(path):
    with open(path, 'r') as f:
        return json.load(f)

# Generate instances using py3dbp
def gen_py3dbp_instances(
    num_instances,
    bin_count_range=(2, 5),
    bin_length_range=(800, 1600),
    bin_width_range=(600, 1200),
    bin_depth_range=(400, 1000),
    item_count_range=(5, 15),
    item_frac_range=(0.05, 0.5),
    weight_range=(0.5, 5.0),
    seed=None
):
    if seed is not None:
        random.seed(seed)

    instances = []
    for inst_idx in range(num_instances):
        packer = Packer()
        # Create bins
        n_bins = random.randint(*bin_count_range)
        for i in range(n_bins):
            bl = random.randint(*bin_length_range)
            bw = random.randint(*bin_width_range)
            bd = random.randint(*bin_depth_range)
            max_w = weight_range[1] * item_count_range[1] * 1.2
            packer.add_bin(Bin(f"Bin{i}", bl, bw, bd, max_w))
        # Create items
        for b_idx, b in enumerate(packer.bins):
            n_items = random.randint(*item_count_range)
            for j in range(n_items):
                l = int(random.uniform(*item_frac_range) * b.width)
                w = int(random.uniform(*item_frac_range) * b.height)
                d = int(random.uniform(*item_frac_range) * b.depth)
                wt = random.uniform(*weight_range)
                packer.add_item(Item(f"Item{b_idx}_{j}", l, w, d, wt))
        # Convert to dict format
        bins = [
            {"id": idx, "length": b.width, "width": b.height, "depth": b.depth}
            for idx, b in enumerate(packer.bins)
        ]
        boxes = [
            {"id": item.name, "length": item.width, "width": item.height, "depth": item.depth}
            for item in packer.items
        ]
        instances.append({"bins": bins, "boxes": boxes})
    return instances

# Alias for compatibility with my replaced designer method
def gen_random_instances(num): return gen_py3dbp_instances(num)

class OfflinePackingEnv:
    def __init__(self, bins, boxes, grid=10):
        self.bins = bins
        self.boxes = boxes
        self.grid = grid
        self.reset()

    def reset(self):
        self.placed = []
        self.remaining = list(range(len(self.boxes)))
        self.positions = []
        self.last_score = 0.0
        return self._obs()

    def step(self, action):
        if action in self.placed or action >= len(self.boxes):
            done, placed = True, False
        else:
            box = self.boxes[action]
            placed = False
            #not the best practice but readable and it works so it's good enough
            for bin_spec in self.bins:
                for z in range(0, bin_spec["depth"], self.grid):
                    for y in range(0, bin_spec["width"], self.grid):
                        for x in range(0, bin_spec["length"], self.grid):
                            if self._can_place(box, x, y, z, bin_spec):
                                self.placed.append(action)
                                self.remaining.remove(action)
                                self.positions.append((action, x, y, z, bin_spec["id"]))
                                placed = True
                                break
                        if placed: break
                    if placed: break
                if placed: break
            done = not placed

        placed_boxes = len(self.placed)
        total_boxes = len(self.boxes)
        pr = placed_boxes / total_boxes

        used_volume = sum(
        self.boxes[i]["length"] * self.boxes[i]["width"] * self.boxes[i]["depth"]
        for i in self.placed
            )

        # only include bins used
        bins_used = {pos[4] for pos in self.positions}
        total_bin_volume = sum(
        b["length"] * b["width"] * b["depth"]
        for b in self.bins
        if b["id"] in bins_used
        ) or 1.0
        vu = used_volume / total_bin_volume

        # mean waste over only opened bins
        wastes = []
        for b_spec in self.bins:
            if b_spec["id"] not in bins_used:
                continue
            vol_b = b_spec["length"] * b_spec["width"] * b_spec["depth"] or 1.0
            used_b = sum(
                self.boxes[i]["length"] * self.boxes[i]["width"] * self.boxes[i]["depth"]
                for (i, *_, bid) in self.positions if bid == b_spec["id"]
            )
            wastes.append(1.0 - used_b / vol_b)
        mu_w = sum(wastes) / len(wastes) if wastes else 0.0

        α, β, γ, δ = 0.65, 0.25, 0.30, 0.25
        curr_score = α * pr + β * vu - γ * (len(bins_used)/len(self.bins)) - δ * mu_w
        curr_score = max(0.0, min(1.0, curr_score))

        reward = curr_score - self.last_score
        self.last_score = curr_score

        if placed and not self.remaining:
            done = True

        info = {
            "placement_ratio": pr,
            "volume_utilization": vu,
            "bin_penalty": (len(bins_used)/len(self.bins)),
            "mean_bin_waste": mu_w,
            "score": curr_score,
        }

        return self._obs(), reward, done, info

    def _can_place(self, box, x, y, z, bin_spec):
        bx, by, bz = box["length"], box["width"], box["depth"]
        if x + bx > bin_spec["length"] or y + by > bin_spec["width"] or z + bz > bin_spec["depth"]:
            return False
        for idx, ox, oy, oz, b_id in self.positions:
            if b_id != bin_spec["id"]:
                continue
            if self._overlaps(x, y, z, bx, by, bz, ox, oy, oz, self.boxes[idx]):
                return False
        return True

    def _overlaps(self, x1, y1, z1, dx1, dy1, dz1, x2, y2, z2, box2):
        dx2, dy2, dz2 = box2["length"], box2["width"], box2["depth"]
        return not (
            x1 + dx1 <= x2 or x2 + dx2 <= x1 or
            y1 + dy1 <= y2 or y2 + dy2 <= y1 or
            z1 + dz1 <= z2 or z2 + dz2 <= z1
        )

    def _obs(self):
        mask = np.zeros(len(self.boxes), dtype=np.float32)
        for idx in self.placed:
            mask[idx] = 1.0
        specs = np.array([[b["length"], b["width"], b["depth"]] for b in self.boxes], dtype=np.float32)
        return {"placed_mask": mask, "box_specs": specs}

    def render(self):
        print("Placed:", [self.boxes[i]["id"] for i in self.placed])

class SimplePPOPolicy(nn.Module):
    def __init__(self, feature_dim=4, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2), nn.ReLU()
        )
        # Actor & critic heads 
        self.actor = nn.Linear(hidden_dim//2, 1)
        self.critic = nn.Linear(hidden_dim//2, 1)

    def forward(self, box_specs, mask):
        batch, n, _ = box_specs.shape
        features = torch.cat([box_specs, mask.unsqueeze(-1)], dim=2)
        x = self.shared(features)
        logits = self.actor(x).squeeze(-1).masked_fill(mask == 1, -1e9)
        value = self.critic(x).mean(dim=1).squeeze(-1)
        return logits, value

if __name__ == "__main__":
    single_paths = list(SINGLE_JSON_DIR.glob("*.json"))
    if not single_paths:
        logger.error("No single-bin JSONs found")
        sys.exit(1)

    instances = [load_instance(p) for p in single_paths] + gen_py3dbp_instances(1400)
    policy = optimizer = None
    reward_curve, comp_logs = [], []
    start_time = time.perf_counter()
    logger.info("Training PPO with policy gradient across multiple instances")

    for episode in range(EPISODES):
        inst = random.choice(instances)
        bins = inst["bins"]
        boxes = [b for b in inst["boxes"] if any(
            b["length"] <= bn["length"] and b["width"] <= bn["width"] and b["depth"] <= bn["depth"]
            for bn in bins
        )]
        if not boxes:
            continue

        env = OfflinePackingEnv(bins, boxes)
        obs = env.reset()

        if policy is None:
            policy = SimplePPOPolicy()
            optimizer = optim.Adam(policy.parameters(), lr=1e-3)

        done = False
        total_reward = 0
        log_probs = []
        values = []
        rewards = []
        comp_info = {}

        while not done:
            box_specs = torch.tensor(obs["box_specs"]).unsqueeze(0)
            mask = torch.tensor(obs["placed_mask"]).unsqueeze(0)
            logits, value = policy(box_specs, mask)
            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            obs, reward, done, info = env.step(action.item())
            total_reward += reward
            comp_info = info
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        values = torch.cat(values)
        advantages = returns - values.detach().squeeze()

        log_probs = torch.stack(log_probs)
        old_log_probs = log_probs.detach()
        ratios = torch.exp(log_probs - old_log_probs)
        clip_eps = 0.1
        clipped = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(ratios * advantages, clipped).mean()
        value_loss = (values.squeeze() - returns).pow(2).mean()
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode == EPISODES // 2 - 1:
            for pg in optimizer.param_groups:
                pg['lr'] /= 2
            logger.info(f"Annealed learning rate to {optimizer.param_groups[0]['lr']:.6f}")

        reward_curve.append(total_reward)
        comp_logs.append(comp_info)
        logger.info(
            f"Episode {episode} total reward: {total_reward:.4f} | components: "
            f"pr={comp_info.get('placement_ratio', 0.0):.3f}, vu={comp_info.get('volume_utilization', 0.0):.3f}, "
            f"bp={comp_info.get('bin_penalty', 0.0):.3f}, mu_w={comp_info.get('mean_bin_waste', 0.0):.3f}")

    if comp_logs:
        keys = comp_logs[0].keys()
        with open(COMP_LOG_CSV, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(comp_logs)

    duration = time.perf_counter() - start_time
    logger.info(f"Training completed in {duration:.2f} seconds")
    torch.save(policy.state_dict(), MODEL_OUTPUT)
    logger.info(f"Saved policy model to {MODEL_OUTPUT}")

    # reward curve simple
    plt.figure(figsize=(50, 24), dpi=350)
    plt.plot(reward_curve)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("PPO Training Curve")
    plt.grid(True)
    plt.savefig(f"ppo_reward_curve_{timestamp}.png")
    plt.show()
