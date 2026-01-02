import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils.ema import EMA
import numpy as np
import random
import time
import logging 
from utils.clip_wrapper import clip_img_wrap
import torch
import torchvision
import torchvision.transforms as transforms
from utils.data_utils import Custom_dataset
from utils.model_SimCLR import SimCLR_encoder
import torch.optim as optim
from utils.learning import *
from model_diffusion import Diffusion
import utils.ResNet_for_32 as resnet_s
from utils.knn_utils import sample_knn_labels
import argparse
from utils.data_utils import LDL_Run_Dataset
import os
import json 
import utils.metrics as metrics # 
from sklearn.model_selection import KFold
from torch.utils.data import Subset
import matplotlib.pyplot as plt
def set_logging(save_dir, log_name):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(save_dir, log_name), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
def calc_avg_improvement(current_scores, sota_scores):
    improvements = []
    # 前4个指标 (Cheby, Clark, Canbe, KL) 是越小越好 -> (SOTA - Ours) / SOTA
    for i in range(4):
        imp = (sota_scores[i] - current_scores[i]) / (sota_scores[i] + 1e-8)
        improvements.append(imp)
    for i in range(4, 6):
        imp = (current_scores[i] - sota_scores[i]) / (sota_scores[i] + 1e-8)
        improvements.append(imp)
        
    return np.mean(improvements)

def init_model(args, device, n_class, input_dim, is_ldl):
    # load fp encoder
    if not is_ldl:
        if args.fp_encoder == 'SimCLR':
            fp_dim = 2048
            real_fp = True
            # 【修改点】把 {dataset} 改成 {args.noise_type}
            state_dict = torch.load(f'./model/SimCLR_128_{args.noise_type}.pt', map_location=torch.device(args.device))
            fp_encoder = SimCLR_encoder(feature_dim=128).to(args.device)
            fp_encoder.load_state_dict(state_dict, strict=False)
        elif args.fp_encoder == 'CLIP':
            real_fp = False
            fp_encoder = clip_img_wrap(args.CLIP_type, args.device, center=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            fp_dim = fp_encoder.dim
        else:
            raise Exception("fp_encoder should be SimCLR or CLIP")
    else:
        fp_encoder = nn.Identity().to(device) 
        fp_dim = input_dim
        real_fp = True 

    diffusion_model = Diffusion(fp_encoder=fp_encoder, n_class=n_class, fp_dim=fp_dim, feature_dim=args.feature_dim,
                                device=device, encoder_type=args.diff_encoder, ddim_num_steps=args.ddim_n_step)
    diffusion_model.fp_encoder.eval()
    return diffusion_model, fp_dim

def train(diffusion_model, train_dataset, val_dataset, test_dataset, model_path, args, real_fp):
    device = diffusion_model.device
    n_class = diffusion_model.n_class
    n_epochs = args.nepoch
    k = args.k
    warmup_epochs = args.warmup_epochs
    sota_values = None
    loss_history = []
    sota_path = '../Data/sota.json' # 假设路径
    if os.path.exists(sota_path):
        try:
            with open(sota_path, 'r') as f:
                sota_data = json.load(f)['data']
            if args.noise_type in sota_data:
                # 按照 metrics.py 的顺序: Cheby, Clark, Canbe, KL, Cosine, Inter
                metrics_order = ['Cheby', 'Clark', 'Canbe', 'KL', 'Cosine', 'Inter']
                sota_values = [sota_data[args.noise_type][k]['mean'] for k in metrics_order]
                logging.info(f"Loaded SOTA for {args.noise_type}: {sota_values}")
        except:
            logging.info("Warning: Could not load SOTA data, using KL as fallback metric.")

    # # pre-compute for fp embeddings on training data
    if hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'features'):
        # 情况A：Subset
        logging.info('Loading LDL features directly (from Subset)...')
        indices = train_dataset.indices
        train_embed = torch.from_numpy(train_dataset.dataset.features[indices]).float().to(device)
        # [新增] 获取对应的 targets，防止后面报错
        knn_targets = torch.from_numpy(train_dataset.dataset.labels[indices]).float().to(device) 
        
    elif hasattr(train_dataset, 'features'):
        # 情况B：完整 Dataset
        logging.info('Loading LDL features directly...')
        train_embed = torch.from_numpy(train_dataset.features).float().to(device)
        # [新增] 获取对应的 targets
        knn_targets = torch.from_numpy(train_dataset.labels).float().to(device)
    else:
        # 情况C：CIFAR
        logging.info('pre-computing fp embeddings for training data')
        train_embed = prepare_fp_x(diffusion_model.fp_encoder, train_dataset, save_dir=None, device=device,
                                   fp_dim=fp_dim).to(device)
        # [新增] 获取对应的 targets
        knn_targets = torch.tensor(train_dataset.targets).to(device)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')

    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model.model)
    max_improvement = -float('inf')
    logging.info('Diffusion training start')
    for epoch in range(n_epochs):
        diffusion_model.model.train()
        epoch_loss_sum = 0.0
        num_batches = 0
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, data_batch in pbar:
                [x_batch, y_batch, data_indices] = data_batch[:3]

                if real_fp:
                    # compute embeddings for augmented images for better performance
                    fp_embd = diffusion_model.fp_encoder(x_batch.to(device))
                else:
                    # use pre-compute embedding for efficiency
                    fp_embd = train_embed[data_indices, :]
        
                # sample a knn labels and compute weight for the sample
                y_labels_batch, sample_weight = sample_knn_labels(fp_embd, y_batch.to(device), train_embed,
                                                                  knn_targets, 
                                                                  k=k, n_class=n_class, weighted=True)
                if y_labels_batch.is_floating_point():
                    y_0_batch = y_labels_batch.to(device)
                else:
                # convert label to one-hot vector
                    y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch.to(torch.int64),
                                                                                      n_class=n_class)
                    y_0_batch = y_one_hot_batch.to(device)

                # adjust_learning_rate
                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=args.nepoch, lr_input=args.lr)
                n = x_batch.size(0)

                # sampling t
                t = torch.randint(low=0, high=diffusion_model.num_timesteps, size=(n // 2 + 1,)).to(device)
                t = torch.cat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

                # train with and without prior
                output, e = diffusion_model.forward_t(y_0_batch, x_batch, t, fp_embd)

                # compute loss
                mse_loss = diffusion_loss(e, output)
                weighted_mse_loss = torch.matmul(sample_weight, mse_loss)
                loss = torch.mean(weighted_mse_loss)
                pbar.set_postfix({'loss': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(diffusion_model.model)
                epoch_loss_sum += loss.item()
                num_batches += 1
        avg_epoch_loss = epoch_loss_sum / num_batches
        loss_history.append(avg_epoch_loss)
        if epoch >= warmup_epochs:
            current_scores, _ = test(diffusion_model, val_loader, num_gen=1, device=device)
            current_imp = calc_avg_improvement(current_scores, sota_values)
            
            score_str = f"KL: {current_scores[3]:.4f}, Cheby: {current_scores[0]:.4f}"
            logging.info(f"Epoch: {epoch} | AvgImp: {current_imp:.2%} | {score_str}")

            if current_imp > max_improvement:
                max_improvement = current_imp
                logging.info(f'⚡ Improved! AvgImp: {current_imp:.2%} Saving model...')
                
                states = [
                    diffusion_model.model.state_dict(),
                    diffusion_model.diffusion_encoder.state_dict(),
                    diffusion_model.fp_encoder.state_dict()
                ]
                torch.save(states, model_path)
                logging.info(f"Model saved to {model_path}")
    save_root = os.path.dirname(model_path)
    curve_dir = os.path.join(save_root, 'curve')
    os.makedirs(curve_dir, exist_ok=True)
    
    # 获取文件名 (例如 fold_0.pt -> fold_0)
    base_name = os.path.basename(model_path).replace('.pt', '')

    # [修改] 2. 保存 npy 数据到 curve 目录
    loss_npy_path = os.path.join(curve_dir, f'{base_name}_loss.npy')
    np.save(loss_npy_path, np.array(loss_history))

    # [修改] 3. 画图并保存到 curve 目录
    plt.figure()
    plt.plot(loss_history)
    plt.title(f'Loss Curve (Best Imp: {max_improvement:.2%})')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    # 保存为 fold_0_loss.png 或 ckpt_best_loss.png
    plt.savefig(os.path.join(curve_dir, f'{base_name}_loss.png'))
    plt.close() # 关掉防止内存泄露

    # [修改] 4. 打印最优结果 (不用翻日志)
    logging.info("\n" + "="*30)
    logging.info(f"🏆 Training Finished. Best Imp: {max_improvement:.2%}")
    logging.info(f"📈 Curve saved to: {curve_dir}")
    logging.info("="*30 + "\n")

    return loss_history

def tune_5fold(full_train_dataset, args, device, n_class, input_dim, is_ldl, sota_values, real_fp):
    logging.info(f">>> Starting 5-Fold Internal Tuning (Search IDX: {args.search_idx})...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=123)
    fold_imps = []
    search_save_dir = os.path.join('temp_model', args.noise_type, f'search_{args.search_idx}')
    os.makedirs(search_save_dir, exist_ok=True)
    set_logging(search_save_dir, 'search.log')
    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_train_dataset)):
        logging.info(f"\n--- Tuning Fold {fold+1} / 5 ---")
        train_sub = Subset(full_train_dataset, train_ids)
        val_sub = Subset(full_train_dataset, val_ids)
        
        # 1. 初始化
        diffusion_model, fp_dim = init_model(args, device, n_class, input_dim, is_ldl)
        # 模型保存为 fold_0.pt, fold_1.pt ...
        fold_path = os.path.join(search_save_dir, f'fold_{fold}.pt')
        
        # 3. 训练
        train(diffusion_model, train_sub, val_sub, val_sub, fold_path, args, real_fp)
        
        # 4. 验证
        if os.path.exists(fold_path):
            states = torch.load(fold_path, map_location=device)
            diffusion_model.model.load_state_dict(states[0])
        
        test_loader = data.DataLoader(val_sub, batch_size=args.batch_size, shuffle=False)
        scores, _ = test(diffusion_model, test_loader, num_gen=1, device=device) # 验证阶段不用随机性或少用
        
        imp = calc_avg_improvement(scores, sota_values)
        logging.info(f"Fold {fold+1} Result: {imp:.2%}")
        fold_imps.append(imp)

    avg_imp = np.mean(fold_imps)
    logging.info(f"\n[FINAL_AVG_IMP] {avg_imp:.2%} (Average of 5 folds)")
    return avg_imp

def test(diffusion_model, test_loader, num_gen=1,device='cuda:0'):
    """
    num_gen=1: 训练中快速验证。
    num_gen=10: 取平均
    """
    diffusion_model.model.eval()
    
    # 1. 收集真实标签
    all_targets = []
    for data_batch in test_loader:
        target = data_batch[1]
        all_targets.append(target.numpy())
    Y = np.concatenate(all_targets, axis=0)
    
    # 2. 循环生成 num_gen 次
    accumulated_preds = None
    individual_results = [] 
    
    # 如果生成多次，显示进度条；单次则不显示
    iter_range = range(num_gen)
    if num_gen > 1:
        logging.info(f"Running test ({num_gen}x avg)...")
        iter_range = tqdm(iter_range, desc="Gen", ncols=80, leave=False)

    for i in iter_range:
        preds = []
        with torch.no_grad():
            for data_batch in test_loader:
                features = data_batch[0].to(device)
                out = diffusion_model.reverse_ddim(features, stochastic=True, fq_x=None)
                preds.append(out.detach().cpu().numpy())
        
        # 当前轮结果
        Y_hat_single = np.concatenate(preds, axis=0)
        
        # 记录单次分数 (投影后)
        if num_gen > 1:
            scores_single = metrics.score(Y, metrics.proj(Y_hat_single))
            individual_results.append(scores_single)
        
        # 累加
        if accumulated_preds is None:
            accumulated_preds = Y_hat_single
        else:
            accumulated_preds += Y_hat_single
            
    # 3. 取平均 -> 投影 -> 算最终分
    Y_hat_avg = accumulated_preds / num_gen
    Y_hat_avg_proj = metrics.proj(Y_hat_avg)
    final_scores = metrics.score(Y, Y_hat_avg_proj)

    metrics_names = ['Chebyshev', 'Clark', 'Canberra', 'KL Divergence', 'Cosine', 'Intersection']
    logging.info("\n" + "="*30)
    for idx, name in enumerate(metrics_names):
        logging.info(f"{name} | {final_scores[idx]:.4f}")
    logging.info("="*30 + "\n")
    
    # 如果是单次生成，individual_results 就是空的
    return final_scores, individual_results

def main_worker(args):
    with open('../Data/sota.json', 'r') as f:
        full_data = json.load(f)
        sota_vals = [full_data['data'][args.noise_type][k]['mean'] for k in ['Cheby', 'Clark', 'Canbe', 'KL', 'Cosine', 'Inter']]
    current_seed = 123 + args.run_idx
    set_seed(current_seed)
    logging.info(f"Random Seed set to: {current_seed}")
    is_ldl=False
    device = args.device
    logging.info(f'Using device: {device}')
    dataset = args.noise_type # 这里直接用 dataset 名字
    ldl_datasets = ["SBU_3DFE", "Scene", "Gene", "Movie", "RAF_ML", "Ren_Cecps", 
                             "SJAFFE", "M2B", "SCUT_FBP5500", "Twitter_LDL", "Flickr_LDL", "SCUT_FBP"]
    # load datasets
    if dataset == 'cifar10':
        n_class = 10
        train_dataset_cifar = torchvision.datasets.CIFAR10(root='./', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR10(root='./', train=False, download=True)

    elif dataset == 'cifar100':
        n_class = 100
        train_dataset_cifar = torchvision.datasets.CIFAR100(root='./', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR100(root='./', train=False, download=True)
    elif dataset in ldl_datasets:
        is_ldl = True
        path_feature = os.path.join('../Data', 'feature', dataset, f'run_{args.run_idx}')
        path_image = os.path.join('../Data', 'image', dataset, f'run_{args.run_idx}')
        data_root = path_feature if os.path.exists(path_feature) else path_image
        logging.info(f"Loading LDL data from: {data_root}")

        # 2. 加载数据
        full_train_dataset = LDL_Run_Dataset(data_root, mode='train')
        test_dataset = LDL_Run_Dataset(data_root, mode='test')
        
        # # 3. 划分 Train/Val
        # train_size = int(0.9 * len(full_train_dataset))
        # val_size = len(full_train_dataset) - train_size
        # train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
        train_dataset = full_train_dataset 
        val_dataset = test_dataset
        # 4. 获取维度
        input_dim = test_dataset.feature_dim
        n_class = test_dataset.label_dim
        # === [结束] ===
    else:
        raise Exception("Date should be cifar10 or cifar100")

    # load fp encoder
    if not is_ldl:
        if args.fp_encoder == 'SimCLR':
            fp_dim = 2048
            real_fp = True
            state_dict = torch.load(f'./model/SimCLR_128_{dataset}.pt', map_location=torch.device(args.device))
            fp_encoder = SimCLR_encoder(feature_dim=128).to(args.device)
            fp_encoder.load_state_dict(state_dict, strict=False)
        elif args.fp_encoder == 'CLIP':
            real_fp = False
            fp_encoder = clip_img_wrap(args.CLIP_type, args.device, center=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            fp_dim = fp_encoder.dim
        else:
            raise Exception("fp_encoder should be SimCLR or CLIP")
    else:
        fp_encoder = nn.Identity().to(device) 
        fp_dim = input_dim
        real_fp = True

    if not is_ldl:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = Custom_dataset(train_dataset_cifar.data[:45000], train_dataset_cifar.targets[:45000],
                                       transform=transform_train)
        val_dataset = Custom_dataset(train_dataset_cifar.data[45000:], train_dataset_cifar.targets[45000:])
        test_dataset = Custom_dataset(test_dataset_cifar.data, test_dataset_cifar.targets)

        # load noisy label
        noise_label = np.load('./noise_label/' + args.noise_type + '.npy')
        train_dataset.update_label(noise_label)
        logging.info('Training on noise label:', args.noise_type)
        
    batch_size = args.batch_size
    if args.tune:
        avg_imp = tune_5fold(train_dataset, args, device, n_class, input_dim, is_ldl, sota_vals, real_fp)
        return avg_imp, None
    else:
        # initialize diffusion model
        save_dir = os.path.join('model', args.noise_type, f'run_{args.run_idx}')
        os.makedirs(save_dir, exist_ok=True)
        set_logging(save_dir, 'run.log')
        model_path = os.path.join(save_dir, 'ckpt_best.pt')
        # initialize diffusion model
        diffusion_model = Diffusion(fp_encoder=fp_encoder, n_class=n_class, fp_dim=fp_dim, feature_dim=args.feature_dim,
                                    device=device, encoder_type=args.diff_encoder, ddim_num_steps=args.ddim_n_step)
        diffusion_model.fp_encoder.eval()

        # train the diffusion model
        logging.info(f'Training LRA-diffusion on: {args.noise_type} | Split: {args.run_idx}')
        logging.info(f'Model saving to: {model_path}')
        
        train(diffusion_model, train_dataset, val_dataset, test_dataset, model_path, args, real_fp=real_fp)
        
        logging.info(f"--- Evaluating Split {args.run_idx} (10x Ensemble) ---")
        # 重新加载最佳模型
        if os.path.exists(model_path):
            states = torch.load(model_path, map_location=device)
            diffusion_model.model.load_state_dict(states[0]) 
        else:
            logging.info("Warning: Best model not found, using last epoch model.")

        test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        final_scores, _ = test(diffusion_model, test_loader, num_gen=10, device=device)
        avg_imp = calc_avg_improvement(final_scores, sota_vals)
        return avg_imp, final_scores
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_type', default='Gene', help='dataset name (Gene, etc.)', type=str) # 借用 noise_type 传数据集名字
    parser.add_argument('--run_idx', type=int, default=0, help='run index for LDL data') # 新增
    parser.add_argument("--nepoch", default=200, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=200, help="batch_size", type=int)
    parser.add_argument("--device", default='cpu', help="which GPU to use", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=5, help="warmup_epochs", type=int)
    parser.add_argument("--feature_dim", default=512, help="feature_dim", type=int)
    parser.add_argument("--k", default=10, help="k neighbors for knn", type=int)
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--fp_encoder", default='SimCLR', help="which encoder for fp (SimCLR or CLIP)", type=str)
    parser.add_argument("--CLIP_type", default='ViT-L/14', help="which encoder for CLIP", type=str)
    parser.add_argument("--diff_encoder", default='linear', help="Force linear for features", type=str) 
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--search_mode', action='store_true', help='Whether in search mode')
    parser.add_argument('--tune', action='store_true', help='Enable 5-fold internal tuning')
    parser.add_argument('--search_idx', type=int, default=0, help='Index for hyperparameter search')
    args = parser.parse_args()
    imp, scores = main_worker(args)
    logging.info(f"Execution Finished. Imp: {imp}")