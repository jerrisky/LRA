import os, json, argparse, subprocess, itertools, sys, re
import numpy as np

# 配置
TRAIN_SCRIPT = "train_LDL_Features.py"
RESULT_ROOT = "result"
METRICS_NAMES = ['Chebyshev', 'Clark', 'Canberra', 'KL Divergence', 'Cosine', 'Intersection']

SEARCH_SPACE = {
    "lr": [1e-4, 5e-5],
    "feature_dim": [512],
    "k": [10, 20, 50]  # 搜索不同的邻居数
}

def run_cmd_live(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    full_output = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None: break
        if line:
            sys.stdout.write(line)
            full_output.append(line)
    return "".join(full_output)

def parse_metrics(output_str):
    results = []
    for m in METRICS_NAMES:
        pattern = re.escape(m) + r"\s+\|\s+([0-9\.]+)"
        match = re.search(pattern, output_str)
        if match: results.append(float(match.group(1)))
    return results if len(results) == 6 else None

def main(dataset, device):
    save_dir = os.path.join(RESULT_ROOT, dataset)
    os.makedirs(save_dir, exist_ok=True)
    res_txt = os.path.join(save_dir, "result.txt")

    # 1. Grid Search (Split 0) - 使用 200 Epoch
    print("\n🔍 Step 1: Grid Search (Split 0, Epoch: 200)")
    keys, values = zip(*SEARCH_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_kl, best_params = float('inf'), {}

    for params in combinations:
        # 强制设置 --nepoch 200
        cmd = ["python", TRAIN_SCRIPT, "--noise_type", dataset, "--device", device, "--run_idx", "0", "--nepoch", "200"]
        for k, v in params.items(): cmd.extend([f"--{k}", str(v)])
        
        output = run_cmd_live(cmd)
        scores = parse_metrics(output)
        if scores and scores[3] < best_kl:
            best_kl, best_params = scores[3], params

    print(f"\n✅ Best Params Found: {best_params} (KL: {best_kl:.4f})")

    # 2. 10 Splits - 使用 1000 Epoch
    print("\n🏃 Step 2: Running 10 Splits (Epoch: 1000)")
    all_metrics = []
    with open(res_txt, 'w') as f: 
        f.write(f"Dataset: {dataset}\nBest Params: {best_params}\nSearch Epoch: 200 | Final Epoch: 1000\n\n")

    for run_idx in range(10):
        print(f"\n>>> Split {run_idx}/9")
        # 强制设置 --nepoch 1000
        cmd = ["python", TRAIN_SCRIPT, "--noise_type", dataset, "--device", device, "--run_idx", str(run_idx), "--nepoch", "1000"]
        for k, v in best_params.items(): cmd.extend([f"--{k}", str(v)])
        
        output = run_cmd_live(cmd)
        scores = parse_metrics(output)
        if scores:
            all_metrics.append(scores)
            with open(res_txt, 'a') as f:
                f.write(f"Split {run_idx} | KL: {scores[3]:.4f} | Metrics: {scores}\n")

    # 3. 汇总 Mean ± Std
    if all_metrics:
        all_metrics = np.array(all_metrics)
        means, stds = np.mean(all_metrics, axis=0), np.std(all_metrics, axis=0)
        
        summary = "\n" + "="*50 + "\nFinal Aggregation (Mean ± Std, Epoch 1000):\n"
        for i, name in enumerate(METRICS_NAMES):
            summary += f"{name:<15} : {means[i]:.4f} ± {stds[i]:.4f}\n"
        
        print(summary)
        with open(res_txt, 'a') as f: f.write(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    main(args.dataset, args.device)