"""
train.py  —  DFL for FSTSP on Amazon dataset (CARC version)
Direct implementation — no PyEPO multiprocess pool (avoids CARC hang)
Usage:
    python train.py --method PtO  --epochs 20
    python train.py --method DBB  --lambd 5 --epochs 10
    python train.py --method RS   --sigma 1.0 --epochs 10
    python train.py --method PGB  --H 0.05 --epochs 10
    python train.py --method PGC  --H 0.05 --epochs 10
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
import os, subprocess, random, time, pickle, json, argparse
import itertools
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader

# ── Parse arguments ───────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, required=True,
                    choices=['PtO','DBB','RS','PGB','PGC','SPO','PFYL'])
parser.add_argument('--lambd',  type=float, default=5.0)
parser.add_argument('--sigma',  type=float, default=1.0,
                    help='RS/PFYL: noise amplitude sigma')
parser.add_argument('--h',      type=float, default=0.05,
                    help='PGB/PGC: finite difference step size h')
parser.add_argument('--epochs', type=int,   default=10)
parser.add_argument('--lr',     type=float, default=1e-3)
parser.add_argument('--batch',  type=int,   default=8)
parser.add_argument('--subset', type=int,   default=0,
                    help='Use first N training instances (0=all)')
                    
parser.add_argument('--output_dir', type=str, default='saved_models')
parser.add_argument('--init_from', type=str, default='',
                    help='Path to pretrained model for warm start')

args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT      = os.path.expanduser("~/1DFL")
AMAZON_DIR   = f"{PROJECT}/amazon_instances"
CONFIG_TRAIN = f"{AMAZON_DIR}/configs_train"
CONFIG_VAL   = f"{AMAZON_DIR}/configs_val"
CONFIG_TEST  = f"{AMAZON_DIR}/configs_test"
DATASET_DIR  = f"{AMAZON_DIR}/dataset"
MODEL_DIR    = f"{PROJECT}/{args.output_dir}" # input output dir name
# MODEL_DIR    = f"{PROJECT}/saved_models4"  # modify
WORK_DIR     = f"/tmp/fstsp_{args.method}_{os.getpid()}" 
LKH_PATH     = f"{PROJECT}/LKH"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(WORK_DIR,  exist_ok=True)
os.chmod(LKH_PATH, 0o755)

# ── Constants ─────────────────────────────────────────────────────────
n          = 9
n_orig     = n + 2
SCALE      = 10
BATCH_SIZE = args.batch
NUM_EPOCHS = args.epochs
LR         = args.lr
LAMBD      = args.lambd
SIGMA      = args.sigma   # RS/PFYL noise amplitude
H          = args.h       # PGB/PGC step size

# ── Globals for config tracking ───────────────────────────────────────
_current_config_dir  = None
_current_config_name = None
_batch_configs       = []
_solve_id            = itertools.count()


# ── Helper functions ──────────────────────────────────────────────────

def load_structure_amazon(tspmd_path):
    node_to_group={1:0}; groups={}; fake_to_orig={}; svc_base={}
    in_ctsp=in_svc=in_draft=False; N_total=0
    with open(tspmd_path) as f:
        for line in f:
            if "DIMENSION" in line and ":" in line:
                try: N_total=int(line.split(":")[1])
                except: pass
            if "CTSP_SET_SECTION"     in line: in_ctsp=True;  in_svc=in_draft=False; continue
            if "SERVICE_TIME_SECTION" in line: in_svc=True;   in_ctsp=in_draft=False; continue
            if "DRAFT_LIMIT_SECTION"  in line: in_draft=True; in_svc=in_ctsp=False;  continue
            if "DEPOT_SECTION"        in line: break
            if in_ctsp:
                p=line.split()
                if p and p[-1]=="-1":
                    g=int(p[0]); nodes=[int(x) for x in p[1:-1]]
                    groups[g]=nodes
                    for nd in nodes: node_to_group[nd]=g
            if in_svc:
                p=line.split()
                if len(p)==2: svc_base[int(p[0])]=float(p[1])
            if in_draft:
                p=line.split()
                if len(p)==2:
                    fake_to_orig[int(p[0])]=int(p[1])
    saved_colors={}
    for nd in range(1, N_total+1):
        g=node_to_group.get(nd,0)
        saved_colors[nd]=-g if (g>0 and svc_base.get(nd,1.0)==0.0) else g
    ctsp_lines=[]; draft_lines=[]
    in_ctsp=in_draft=False
    with open(tspmd_path) as f:
        for line in f:
            if "CTSP_SET_SECTION"     in line: in_ctsp=True;  in_draft=False; ctsp_lines.append(line); continue
            if "DRAFT_LIMIT_SECTION"  in line: in_draft=True; in_ctsp=False;  draft_lines.append(line); continue
            if "SERVICE_TIME_SECTION" in line or "DEPOT_SECTION" in line: in_ctsp=in_draft=False
            if in_ctsp:  ctsp_lines.append(line)
            if in_draft: draft_lines.append(line)
    return dict(N_total=N_total, node_to_group=node_to_group,
                groups=groups, fake_to_orig=fake_to_orig,
                saved_colors=saved_colors, ctsp_lines=ctsp_lines,
                draft_lines=draft_lines)


def z_star_amazon(t_hat, struct, config_name, suffix="pred"):
    N = struct['N_total']
    out_tspmd   = Path(WORK_DIR) / f"{config_name}_{suffix}.tspmd"
    out_outtour = Path(WORK_DIR) / f"{config_name}_{suffix}.outtour"
    out_par     = Path(WORK_DIR) / f"{config_name}_{suffix}.par"

    tau_truck = t_hat[:n_orig**2].reshape(n_orig, n_orig)
    tau_drone = t_hat[n_orig**2:].reshape(n_orig, n_orig)

    W_int = np.zeros((N, N), dtype=np.int64)
    svc   = np.zeros(N, dtype=np.int64)
    for a in range(1, N+1):
        oa = struct['fake_to_orig'][a]
        for b in range(1, N+1):
            W_int[a-1,b-1] = round(SCALE * tau_truck[oa, struct['fake_to_orig'][b]])
    for g, nodes in struct['groups'].items():
        for nd in nodes:
            svc[nd-1] = round(SCALE * tau_drone[struct['fake_to_orig'][nd], g])

    with open(out_tspmd, 'w') as f:
        f.write(f"NAME : {config_name}_{suffix}\nCOMMENT : DFL\n")
        f.write(f"TYPE : TSPMD\nDIMENSION : {N}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("DRONES : 1\nEDGE_WEIGHT_SECTION\n")
        for row in W_int: f.write(" ".join(map(str, row))+"\n")
        f.writelines(struct['ctsp_lines'])
        f.write("SERVICE_TIME_SECTION\n")
        for i in range(N): f.write(f"{i+1} {svc[i]}\n")
        f.writelines(struct['draft_lines'])
        f.write("DEPOT_SECTION\n1\n-1\nEOF\n")

    with open(out_par, 'w') as f:
        f.write("SPECIAL\n")
        f.write(f"PROBLEM_FILE = {out_tspmd.resolve()}\n")
        f.write("MAX_TRIALS = 1000\nRUNS = 1\nTIME_LIMIT = 10\n")
        f.write(f"TOUR_FILE = {out_outtour.resolve()}\n")

    subprocess.run([LKH_PATH, str(out_par)], capture_output=True,
                   check=True, timeout=120)
    return decode_outtour_amazon(str(out_outtour), struct)


def decode_outtour_amazon(outtour_path, struct):
    tour_all=[]; in_tour=False
    with open(outtour_path) as f:
        for line in f:
            if "TOUR_SECTION" in line: in_tour=True; continue
            if in_tour:
                v=line.strip()
                if v in("-1","EOF"): break
                nd=int(v)
                if 1<=nd<=struct['N_total']: tour_all.append(nd)
    color_count=[0]*(n+1); truck_route=[]; group_launch={}
    for nd in tour_all:
        truck_route.append(nd)
        g=struct['node_to_group'].get(nd,0)
        if g>0:
            sc=struct['saved_colors'].get(nd,g)
            if sc<0: color_count[g]=2
            else:
                color_count[g]+=1
                if color_count[g]==1: group_launch[g]=nd
        if all(color_count[g]>=2 for g in range(1,n+1)): break
    truck_route.append(truck_route[0])
    z_truck=np.zeros((n_orig,n_orig)); z_drone=np.zeros((n_orig,n_orig))
    for i in range(len(truck_route)-1):
        a,b=truck_route[i],truck_route[i+1]
        z_truck[struct['fake_to_orig'][a], struct['fake_to_orig'][b]]+=1
    c2=[0]*(n+1); gl={}
    for nd in truck_route[:-1]:
        g=struct['node_to_group'].get(nd,0)
        if g==0: continue
        sc=struct['saved_colors'].get(nd,g)
        if sc<0: c2[g]=2; continue
        c2[g]+=1
        if c2[g]==1: gl[g]=nd
        elif c2[g]==2:
            z_drone[struct['fake_to_orig'][gl[g]],g]+=1
            z_drone[struct['fake_to_orig'][nd],   g]+=1
    return np.concatenate([z_truck.flatten(),
                           z_drone.flatten()]).astype(np.float32)


def solve_one(t_hat_np, config_name, config_dir, suffix="pred"):
    """Solve one instance directly — no PyEPO pool."""
    t_hat_np = np.maximum(t_hat_np.astype(np.float64), 1e-6)
    tspmd_path = f"{config_dir}/{config_name}.tspmd"
    struct = load_structure_amazon(tspmd_path)
    z_hat  = z_star_amazon(t_hat_np, struct, config_name, suffix=suffix)
    return z_hat, float(np.dot(t_hat_np, z_hat))


def solve_batch(t_hat_tensor, config_list, config_dir, suffix_prefix="s"):
    """Solve a batch of instances, one by one."""
    B = t_hat_tensor.shape[0]
    z_hats = []
    for i in range(B):
        cfg  = config_list[i % len(config_list)]
        sfx  = f"{suffix_prefix}_{next(_solve_id)}"
        t_np = t_hat_tensor[i].detach().numpy()
        z, _ = solve_one(t_np, cfg, config_dir, suffix=sfx)
        z_hats.append(z)
    return torch.FloatTensor(np.array(z_hats))


# ── PrecomputedOptDataset ─────────────────────────────────────────────

class PrecomputedOptDataset(torch.utils.data.Dataset):
    def __init__(self, S_hat, W_hat, z_hat_oracle, z_oracle):
        self.S_hat        = torch.FloatTensor(S_hat)
        self.W_hat        = torch.FloatTensor(W_hat)
        self.z_hat_oracle = torch.FloatTensor(z_hat_oracle)
        self.z_oracle     = torch.FloatTensor(z_oracle).view(-1,1)
    def __len__(self): return len(self.S_hat)
    def __getitem__(self, idx):
        return (self.S_hat[idx], self.W_hat[idx],
                self.z_hat_oracle[idx], self.z_oracle[idx])


# ── Predictor g_theta ─────────────────────────────────────────────────

class g_theta(nn.Module):
    def __init__(self, dim_s, dim_t, hidden=128, depth=2):
        super().__init__()
        dims=[dim_s]+[hidden]*depth+[dim_t]
        layers=[]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i],dims[i+1]))
            if i<len(dims)-2: layers.append(nn.ReLU())
        layers.append(nn.Softplus())
        self.net=nn.Sequential(*layers)
    def forward(self, s): return self.net(s)


# ── Loss functions (direct, no PyEPO pool) ────────────────────────────

def make_loss_func(method, batch_configs, config_dir):
    """Create loss function for one batch — configs known at call time."""

    if method == 'PtO':
        def loss_func(t, w, zh, z):
            return nn.MSELoss()(t, w)

    # first, get grad_DFL = f(t_hat, w_hat)   # approximate ∂Regret/∂t_hat via LKH-3
    # second, return (grad_DFL * t_hat).sum()
    elif method == 'DBB':
        # Algorithm: z_lambda = z*(t_hat + lambda*w_hat)
        # grad = -(1/lambda)(z_hat - z_lambda)
        def loss_func(t, w, zh, z):
            z_hat  = solve_batch(t.detach(),
                                  batch_configs, config_dir, 'dbb_f') # forward solve z*(t̂)
            t_pert = (t + LAMBD * w).detach()       # t + lambda * w_hat
            z_pert = solve_batch(t_pert,
                                  batch_configs, config_dir, 'dbb_b') # backward solve z*(t̂+λŵ)
            grad   = (z_pert - z_hat) / LAMBD      # -(z_hat - z_lambda)/lambda
            return (grad.detach() * t).sum(1).mean()

    elif method == 'RS':
        # Algorithm: grad = (1/M*sigma) * sum_m (w^T z^m) * xi^m
        N_SAMPLES = 3
        def loss_func(t, w, zh, z):
            grad = torch.zeros_like(t)
            for _ in range(N_SAMPLES):
                xi  = torch.randn_like(t)
                z_m = solve_batch((t + SIGMA * xi).detach(),
                                   batch_configs, config_dir, 'rs') # perturbed solve z*(t̂+σξ)
                obj_m = (w.detach() * z_m).sum(1, keepdim=True)    # w^T z^m
                grad  = grad + obj_m * xi                          # scalar * xi
            grad = grad / (N_SAMPLES * SIGMA)
            return (grad.detach() * t).sum(1).mean()

    elif method == 'PGB':
        # Algorithm: z^- = z*(t_hat - h*w_hat), grad = (z_hat - z^-)/h
        def loss_func(t, w, zh, z):
            z_hat = solve_batch(t.detach(),
                                 batch_configs, config_dir, 'pgb_0') # baseline solve z*(t̂)
            z_bwd = solve_batch((t - H * w).detach(),
                                 batch_configs, config_dir, 'pgb_m') # backward solve z*(t̂-hŵ)
            grad  = (z_hat - z_bwd) / H          # (z_hat - z^-)/h
            return (grad.detach() * t).sum(1).mean()

    elif method == 'PGC':
        def loss_func(t, w, zh, z):
            z_fwd = solve_batch((t + H * w).detach(),
                                 batch_configs, config_dir, 'pgc_p') # forward solve z*(t̂+hŵ)
            z_bwd = solve_batch((t - H * w).detach(),
                                 batch_configs, config_dir, 'pgc_n') # backward solve z*(t̂-hŵ)
            grad = (z_fwd - z_bwd) / (2 * H)
            return (grad.detach() * t).sum(1).mean()
    
    elif method == 'SPO':
        # gradient = 2*z*(w_hat) - 2*z*(2*t_hat - w_hat)
        # z*(w_hat) = z_oracle = zh (precomputed, FREE)
        def loss_func(t, w, zh, z):
            t_spo = torch.clamp(2*t - w, min=1e-6).detach() # 2*t - w can produce negative values
            z_spo = solve_batch(t_spo, batch_configs, config_dir, 'spo') # perturbed solve z*(2t̂-ŵ)
            grad  = 2 * (zh - z_spo)    # zh = z*(w_hat), already in oracle.pkl
            return (grad.detach() * t).sum(1).mean()

    elif method == 'PFYL':
        # Gradient: z*(w_hat) - (1/M) * sum_m z*(t_hat + sigma*epsilon_m)
        # z*(w_hat) = z_oracle = zh (precomputed, FREE)
        N_SAMPLES = 3
        def loss_func(t, w, zh, z):
            z_avg = torch.zeros_like(zh)
            for _ in range(N_SAMPLES):
                xi  = torch.randn_like(t)                     # epsilon_m ~ N(0,I)
                z_m = solve_batch((t + SIGMA * xi).detach(),
                                   batch_configs, config_dir, 'pfyl') # perturbed solve z*(t̂+σξ)
                z_avg = z_avg + z_m                            # accumulate z^m
            z_avg = z_avg / N_SAMPLES                          # (1/M) sum z^m
            # grad = z_oracle_solution - E[z*(t_hat + noise)]
            grad  = zh - z_avg                                 # z*(w_hat) - E[z^m]
            return (grad.detach() * t).sum(1).mean()
            
    return loss_func


# ── Proxy regret (fast, no LKH-3) ────────────────────────────────────

def compute_regret_proxy(predictor, loader):
    predictor.eval()
    total=0.0; n=0
    with torch.no_grad():
        for s_hat, w_hat, z_hat_oracle, z_oracle in loader:
            t_hat       = predictor(s_hat)
            pred_cost   = (w_hat * z_hat_oracle).sum(1)
            oracle_cost = z_oracle.squeeze(1)
            regret = ((pred_cost - oracle_cost) / (oracle_cost.abs() + 1e-8)).mean()
            total += regret.item(); n += 1
    predictor.train()
    return total / max(n, 1)


# ── True regret using LKH-3 ──────────────────────────────────────────

def compute_regret_true(predictor, loader, config_list, config_dir):
    predictor.eval()
    total=0.0; n_inst=0
    print(f"  Computing true regret ({len(config_list)} instances)...",
          flush=True)
    with torch.no_grad():
        for batch_idx, (s_hat, w_hat, z_hat_oracle, z_oracle) in enumerate(loader):
            for i in range(len(s_hat)):
                idx = batch_idx * BATCH_SIZE + i
                if idx >= len(config_list): break
                cfg  = config_list[idx]
                sfx  = f"eval_{next(_solve_id)}"
                t_np = predictor(s_hat[i:i+1]).squeeze().numpy()
                z_hat, _ = solve_one(t_np, cfg, config_dir, suffix=sfx)
                w_i  = w_hat[i].numpy().astype(np.float64)
                z_i  = float(z_oracle[i])
                total += (float(np.dot(w_i, z_hat)) - z_i) / (z_i + 1e-8)
                n_inst += 1
            if n_inst % 100 == 0 and n_inst > 0:
                print(f"  {n_inst}/{len(config_list)} done", flush=True)
    predictor.train()
    return total / max(n_inst, 1)


# ── Training loop ─────────────────────────────────────────────────────
# s_hat — context features (hour, day, coordinates) → shape (8, 26)
# w_hat — realized travel times → shape (8, 242)
# z_hat_oracle — oracle solution z*(w_hat) → shape (8, 242)
# z_oracle — oracle cost w_hat^T z*(w_hat) → shape (8, 1) 
def train_Algorithm1_amazon(predictor, method, config_list, config_dir,
                             loader_tr,
                             loader_ev, eval_configs, eval_config_dir,   # val
                             loader_test_final, test_configs_final,       # test
                             test_config_dir_final,
                             num_epochs=NUM_EPOCHS, lr=LR):
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    predictor.train()
    loss_log=[]; regret_log=[]
    best_regret = float('inf')      # ← track best
    best_epoch  = 0

    print(f"Training {method} for {num_epochs} epochs...", flush=True)

    for epoch in range(num_epochs):
        tick = time.time()

        for batch_idx, (s_hat, w_hat, z_hat_oracle, z_oracle) in enumerate(loader_tr):
            start_idx = batch_idx * BATCH_SIZE
            end_idx   = min(start_idx + BATCH_SIZE, len(config_list))
            batch_cfgs = config_list[start_idx:end_idx]

            # Create loss function with current batch configs
            loss_func = make_loss_func(method, batch_cfgs, config_dir)

            # Step 1 Forward pass
            t_hat = predictor(s_hat) # g_θ(s_hat) → shape (8, 242)
            
            # Step 2 Compute DFL loss, get grad_DFL = ∂Regret/∂t_hat = (z_fwd - z_bwd) / (2*h), and loss = (grad_DFL.detach() * t).sum(), the scalar means nothing just for differentiation
            # loss = [constant_vector grad_DFL.detach()]· t, hence, d(loss)/d(t) = constant_vector = grad_DFL  ✓
            loss  = loss_func(t_hat, w_hat, z_hat_oracle, z_oracle) # Calls LKH-3 to solve FSTSP under predicted costs, computes surrogate loss whose gradient = DFL gradient from Algorithm 1.
            
            # Step 3 Backward pass
            optimizer.zero_grad(); loss.backward() # compute ∂loss/∂θ = (∂t_hat/∂θ)^T · grad_DFL
            nn.utils.clip_grad_norm_(predictor.parameters(), 1.0) # prevent exploding gradients
            optimizer.step() # θ ← θ - lr · ∂loss/∂θ
            loss_log.append(loss.item())

        # TRUE regret on 80 val instances per epoch — standard formula
        # Regret_i = (w^T z*(t_hat) - z*) / z*
        regret = compute_regret_true(predictor, loader_ev,
                                      eval_configs[:80],
                                      eval_config_dir)
        regret_log.append(regret)
        print(f"Epoch {epoch+1:2d}/{num_epochs}  "
              f"loss:{loss.item():9.4f}  "
              f"regret:{regret*100:6.2f}%  "
              f"({time.time()-tick:.1f}s)", flush=True)

        # Save checkpoint after every epoch
        ckpt_path = f"{MODEL_DIR}/ckpt_{method}_ep{epoch+1}.pt"
        torch.save(predictor.state_dict(), ckpt_path)

        # Save best model based on 80-instance regret
        if regret < best_regret:
            best_regret = regret
            best_epoch  = epoch + 1
            best_path   = f"{MODEL_DIR}/amazon_predictor_{method}_best.pt"
            torch.save(predictor.state_dict(), best_path)
            print(f"  → New best at epoch {best_epoch}: "
                  f"{best_regret*100:.2f}%", flush=True)
    
    # Final evaluation using BEST model (not last epoch)
    print(f"\nBest epoch: {best_epoch}, regret={best_regret*100:.2f}%")
    print(f"Loading best model for final evaluation...", flush=True)
    predictor.load_state_dict(torch.load(best_path))

    # Final TRUE regret on 80 test instances
    print(f"\nComputing FINAL true regret on 80 test instances...",
          flush=True)
    true_regret = compute_regret_true(predictor, loader_test_final,
                                       test_configs_final[:80],
                                       test_config_dir_final)
    print(f"FINAL true regret: {true_regret*100:.4f}%", flush=True)

    model_path = f"{MODEL_DIR}/amazon_predictor_{method}.pt"
    torch.save(predictor.state_dict(), model_path)
    log_path = f"{MODEL_DIR}/amazon_regret_log_{method}.json"
    with open(log_path, 'w') as f:
        json.dump({"loss_log": loss_log,
                   "regret_log": regret_log,
                   "true_regret": true_regret,
                   "best_epoch":  best_epoch,
                   "best_regret": best_regret}, f)
    print(f"Saved: {model_path} (best epoch={best_epoch})", flush=True)
    return loss_log, regret_log, true_regret


# ── Load data ─────────────────────────────────────────────────────────

import pandas as pd

with open(f"{DATASET_DIR}/oracle.pkl", 'rb') as f:
    dataset_train, dataset_val, dataset_test = pickle.load(f)

print(f"Oracle loaded: train={len(dataset_train)}, "
      f"val={len(dataset_val)}, test={len(dataset_test)}")

meta_train = pd.read_csv(f"{DATASET_DIR}/meta_train.csv")
meta_val   = pd.read_csv(f"{DATASET_DIR}/meta_val.csv")
meta_test  = pd.read_csv(f"{DATASET_DIR}/meta_test.csv")

train_configs = meta_train['config_name'].tolist()
val_configs   = meta_val['config_name'].tolist() 
test_configs  = meta_test['config_name'].tolist()

# Subset for DFL methods (reduces compute time)
if args.subset > 0:
    train_configs = train_configs[:args.subset]
    dataset_train = torch.utils.data.Subset(dataset_train, range(args.subset))
    print(f'Using subset of {args.subset} training instances')

loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False)
loader_val   = DataLoader(dataset_val,   batch_size=BATCH_SIZE, shuffle=False)
loader_test  = DataLoader(dataset_test,  batch_size=BATCH_SIZE, shuffle=False)

feat_dim = dataset_train[0][0].shape[0]
cost_dim = dataset_train[0][1].shape[0]
print(f"feat_dim={feat_dim}, cost_dim={cost_dim}")
print(f"Method={args.method}, lr={LR}, epochs={NUM_EPOCHS}, "
      f"lambd={LAMBD}, sigma={SIGMA}, h={H}")

# ── Train ─────────────────────────────────────────────────────────────

random.seed(42); np.random.seed(42); torch.manual_seed(42)

predictor = g_theta(dim_s=feat_dim, dim_t=cost_dim)
n_params  = sum(p.numel() for p in predictor.parameters())
print(f"g_theta: {feat_dim} -> {cost_dim}  ({n_params:,} params)")

if args.init_from and os.path.exists(args.init_from):   # warm-start
    predictor.load_state_dict(torch.load(args.init_from, map_location='cpu'))
    print(f"Initialized from: {args.init_from}", flush=True)

ll, rl, true_r = train_Algorithm1_amazon(
    predictor, args.method,
    train_configs, CONFIG_TRAIN,
    loader_train,
    loader_val,        # ← val for per-epoch monitoring
    val_configs,       # ← val configs
    CONFIG_VAL,        # ← val config dir
    loader_test,       # ← test for final evaluation
    test_configs,      # ← test configs
    CONFIG_TEST)       # ← test config dir

print(f"\n{'='*50}")
print(f"Method:             {args.method}")
print(f"Per-epoch regret (80 instances), final epoch: {rl[-1]*100:.4f}%")
print(f"TRUE  test  regret: {true_r*100:.4f}%")
print(f"{'='*50}")
