import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils.ema import EMA
import numpy as np
import random
import time
import logging  # 新增 logging
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
import json # 新增
import utils.metrics as metrics # 新增
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
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

def train(diffusion_model, train_dataset, val_dataset, test_dataset, model_path, args, real_fp):
    device = diffusion_model.device
    n_class = diffusion_model.n_class
    n_epochs = args.nepoch
    k = args.k
    warmup_epochs = args.warmup_epochs
    sota_values = None
    sota_path = '../Data/sota.json' # 假设路径
    if os.path.exists(sota_path):
        try:
            with open(sota_path, 'r') as f:
                sota_data = json.load(f)['data']
            if args.noise_type in sota_data:
                # 按照 metrics.py 的顺序: Cheby, Clark, Canbe, KL, Cosine, Inter
                metrics_order = ['Cheby', 'Clark', 'Canbe', 'KL', 'Cosine', 'Inter']
                sota_values = [sota_data[args.noise_type][k]['mean'] for k in metrics_order]
                print(f"Loaded SOTA for {args.noise_type}: {sota_values}")
        except:
            print("Warning: Could not load SOTA data, using KL as fallback metric.")

    # # pre-compute for fp embeddings on training data
    if hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'features'):
        # 情况A：Subset
        print('Loading LDL features directly (from Subset)...')
        indices = train_dataset.indices
        train_embed = torch.from_numpy(train_dataset.dataset.features[indices]).float().to(device)
        # [新增] 获取对应的 targets，防止后面报错
        knn_targets = torch.from_numpy(train_dataset.dataset.labels[indices]).float().to(device) 
        
    elif hasattr(train_dataset, 'features'):
        # 情况B：完整 Dataset
        print('Loading LDL features directly...')
        train_embed = torch.from_numpy(train_dataset.features).float().to(device)
        # [新增] 获取对应的 targets
        knn_targets = torch.from_numpy(train_dataset.labels).float().to(device)
    else:
        # 情况C：CIFAR
        print('pre-computing fp embeddings for training data')
        train_embed = prepare_fp_x(diffusion_model.fp_encoder, train_dataset, save_dir=None, device=device,
                                   fp_dim=fp_dim).to(device)
        # [新增] 获取对应的 targets
        knn_targets = torch.tensor(train_dataset.targets).to(device)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')

    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model.model)
    max_improvement = -float('inf')
    print('Diffusion training start')
    for epoch in range(n_epochs):
        diffusion_model.model.train()

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
                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=warmup_epochs, n_epochs=1000, lr_input=0.001)
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

        if epoch % 5 == 0 and epoch >= warmup_epochs:
            current_scores, _ = test(diffusion_model, val_loader, num_gen=1, device=device)
            current_imp = calc_avg_improvement(current_scores, sota_values)
            
            score_str = f"KL: {current_scores[3]:.4f}, Cheby: {current_scores[0]:.4f}"
            print(f"Epoch: {epoch} | AvgImp: {current_imp:.2%} | {score_str}")

            if current_imp > max_improvement:
                max_improvement = current_imp
                print(f'⚡ Improved! AvgImp: {current_imp:.2%} Saving model...')
                
                states = [
                    diffusion_model.model.state_dict(),
                    diffusion_model.diffusion_encoder.state_dict(),
                    diffusion_model.fp_encoder.state_dict()
                ]
                torch.save(states, model_path)
                print(f"Model saved to {model_path}")


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
        print(f"Running test ({num_gen}x avg)...")
        iter_range = tqdm(iter_range, desc="Gen", ncols=80, leave=False)

    for i in iter_range:
        preds = []
        with torch.no_grad():
            for data_batch in test_loader:
                features = data_batch[0].to(device)
                out = diffusion_model.reverse_ddim(features, stochastic=False, fq_x=None)
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
    print("\n" + "="*30)
    for idx, name in enumerate(metrics_names):
        print(f"{name} | {final_scores[idx]:.4f}")
    print("="*30 + "\n")
    
    # 如果是单次生成，individual_results 就是空的
    return final_scores, individual_results


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
    parser.add_argument("--k", default=10, help="k neighbors for knn", type=int)
    args = parser.parse_args()
    is_ldl=False
    device = args.device
    print('Using device:', device)
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
        
        # 1. 路径判断 (照搬 DLD)
        path_feature = os.path.join('../Data', 'feature', dataset, f'run_{args.run_idx}')
        path_image = os.path.join('../Data', 'image', dataset, f'run_{args.run_idx}')
        data_root = path_feature if os.path.exists(path_feature) else path_image
        print(f"Loading LDL data from: {data_root}")

        # 2. 加载数据
        full_train_dataset = LDL_Run_Dataset(data_root, mode='train')
        test_dataset = LDL_Run_Dataset(data_root, mode='test')
        
        # 3. 划分 Train/Val
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])

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
        print('Training on noise label:', args.noise_type)
        
    batch_size = args.batch_size

    # initialize diffusion model
    model_path = f'./model/LRA-diffusion_{args.fp_encoder}_{args.noise_type}.pt'
    diffusion_model = Diffusion(fp_encoder=fp_encoder, n_class=n_class, fp_dim=fp_dim, feature_dim=args.feature_dim,
                                device=device, encoder_type=args.diff_encoder, ddim_num_steps=args.ddim_n_step)
    # state_dict = torch.load(model_path, map_location=torch.device(device))
    # diffusion_model.load_diffusion_net(state_dict)
    diffusion_model.fp_encoder.eval()

    # train the diffusion model
    print(f'training LRA-diffusion using fp encoder: {args.fp_encoder} on: {args.noise_type}.')
    print(f'model saving dir: {model_path}')
    train(diffusion_model, train_dataset, val_dataset, test_dataset, model_path, args, real_fp=real_fp)
    logger.info(f"--- Evaluating Split {args.run_idx} (10x Avg) ---")
    diffusion_model.model.load_state_dict(torch.load(model_path, map_location=device))
    final_score, individual_res = test(diffusion_model, test_loader, num_gen=10, device=device)



