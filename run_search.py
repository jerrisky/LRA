import os
import itertools
import numpy as np
import sys
import time
import argparse

# [新增] 直接导入训练脚本
import train_LDL_Features 

# 配置
RESULT_ROOT = "result"
METRICS_NAMES = ['Chebyshev', 'Clark', 'Canberra', 'KL Divergence', 'Cosine', 'Intersection']

# 搜索空间
SEARCH_SPACE = {
    "lr": [1e-3, 5e-4,1e-4],
    "feature_dim": [64,128],
    "k": [10,20],
    "batch_size": [128, 200,256]
}

# 模拟 argparse 解析后的对象
class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def main(dataset, device):
    save_dir = os.path.join(RESULT_ROOT, dataset)
    os.makedirs(save_dir, exist_ok=True)
    res_txt = os.path.join(save_dir, "result.txt")
    
    # 所有日志进这个文件
    final_log_path = os.path.join(save_dir, f"{dataset}.log")

    # 标记运行开始
    with open(final_log_path, 'a') as f:
        f.write(f"\n\n{'='*30}\n")
        f.write(f"🚀 New Execution Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset} | Device: {device}\n")
        f.write(f"{'='*30}\n\n")

    # 基础参数 (对应 train_LDL_Features.py 的默认值)
    base_params = {
        "noise_type": dataset,
        "device": device,
        "run_idx": 0,          
        "search_idx": 0,       # 搜索文件夹索引
        "nepoch": 200,
        "batch_size": 200,
        "num_workers": 4,
        "warmup_epochs": 5,
        "feature_dim": 512,
        "k": 10,
        "ddim_n_step": 10,
        "fp_encoder": 'SimCLR',
        "CLIP_type": 'ViT-L/14',
        "diff_encoder": 'linear',
        "lr": 1e-4,
        "tune": False
    }

    # --- Step 1: Grid Search ---
    print(f"\n🔍 Step 1: Grid Search (Split 0, Epoch: 200) on {device}...")
    
    with open(final_log_path, 'a') as f:
        f.write(f"\n>>> [Step 1] Grid Search Start (Epoch=200)\n")

    keys, values = zip(*SEARCH_SPACE.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_imp = -float('inf')
    best_params = combinations[0]

    for idx, params in enumerate(combinations):
        print(f"👉 Testing Params [{idx}]: {params}")
        with open(final_log_path, 'a') as f:
            f.write(f"\n--- Testing Params [{idx}]: {params} ---\n")

        # 1. 准备参数
        current_args = base_params.copy()
        current_args.update(params)      # 覆盖搜索参数
        current_args['tune'] = True      # 【开启五折】
        current_args['run_idx'] = 0      # 强制只在 Run 0 上搜
        current_args['search_idx'] = idx # 设置搜索目录索引 (search_0, search_1...)
        current_args['nepoch'] = 200     # 搜索阶段 Epoch
        
        # 2. 转为对象
        args_obj = Args(**current_args)
        
        # 3. 直接调用函数
        try:
            avg_imp, _ = train_LDL_Features.main_worker(args_obj)
        except Exception as e:
            print(f"❌ Error in search_{idx}: {e}")
            # 打印详细报错方便调试
            import traceback
            traceback.print_exc()
            avg_imp = -999999999.0

        # 4. 记录结果
        log_str = f"   -> Search_{idx} AvgImp: {avg_imp:.4%}"
        print(log_str)
        with open(final_log_path, 'a') as f: f.write(log_str + "\n")
        
        if avg_imp > best_imp:
            best_imp = avg_imp
            best_params = params
            print("   ⭐ New Best Params!")
            with open(final_log_path, 'a') as f: f.write("   ⭐ New Best Params!\n")

    print(f"\n✅ Search Finished. Best Params: {best_params} (AvgImp: {best_imp:.4%})")

    # --- Step 2: 10 Splits ---
    print("\n🏃 Step 2: Running 10 Splits (Epoch: 2000)")
    
    with open(final_log_path, 'a') as f:
        f.write(f"\n\n>>> [Step 2] Final Training Start (10 Splits, Epoch=2000)\n")
        f.write(f"Best Params: {best_params}\n\n")
    
    all_metrics = []
    
    with open(res_txt, 'w') as f: 
        f.write(f"Dataset: {dataset}\nBest Params: {best_params}\nSearch Best Imp: {best_imp:.4%}\n\n")

    # 准备最佳参数用于正式训练
    final_run_base = base_params.copy()
    final_run_base.update(best_params)
    final_run_base['nepoch'] = 2000 # 正式训练 Epoch
    final_run_base['tune'] = False  # 【关闭五折】
    
    for run_idx in range(1):
        print(f"\n>>> Split {run_idx}/9")
        
        # 更新 run_idx
        final_run_base['run_idx'] = run_idx
        args_obj = Args(**final_run_base)
        
        try:
            _, scores = train_LDL_Features.main_worker(args_obj)
            if scores is not None and len(scores) == 6:
                all_metrics.append(scores)
                res_str = f"Split {run_idx} | " + " | ".join([f"{x:.4f}" for x in scores])
                print(f"✅ {res_str}")
                # 写入结果文件
                with open(res_txt, 'a') as f: f.write(res_str + "\n")
                # 同时也写入详细日志
                with open(final_log_path, 'a') as f: f.write(f"Split {run_idx} Done: {res_str}\n")
            else:
                err_msg = f"❌ Split {run_idx} returned invalid scores."
                print(err_msg)
                with open(final_log_path, 'a') as f: f.write(err_msg + "\n")
                
        except Exception as e:
             err_msg = f"❌ Split {run_idx} Failed: {e}"
             print(err_msg)
             import traceback
             traceback.print_exc()
             with open(final_log_path, 'a') as f: f.write(err_msg + "\n")

    # --- Step 3: Summary ---
    if all_metrics:
        all_metrics = np.array(all_metrics)
        means = np.mean(all_metrics, axis=0)
        stds = np.std(all_metrics, axis=0)
        
        summary = "\n" + "="*50 + "\nFinal Results (Mean ± Std):\n"
        for i, name in enumerate(METRICS_NAMES):
            summary += f"{name:<15} : {means[i]:.4f} ± {stds[i]:.4f}\n"
        
        print(summary)
        with open(res_txt, 'a') as f: f.write(summary)
        # 汇总信息也追加到 log
        with open(final_log_path, 'a') as f: f.write(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    
    main(args.dataset, args.device)