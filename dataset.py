import numpy as np
import torch
from torch.utils.data import Dataset
import os

class DeepONetDataset_E1_1D(Dataset):
    def __init__(self, file_path, quantile=0.95, save_norm_params=True, norm_params_path=None):
        """
        新增：
        - save_norm_params: 是否保存归一化参数到文件
        - norm_params_path: 预加载的归一化参数路径（测试/预测阶段用）
        """
        data = np.load(file_path)
        self.quantile = quantile
        self.norm_params_path = norm_params_path

        # ===================== 1. 加载lnK场和坐标（无修改） =====================
        lnk = torch.from_numpy(data['lnk_field_1d']).float()
        well_locs = data['well_loc_1d']
        N = lnk.shape[0]

        well_mask = torch.zeros((N, 64))
        for i in range(N):
            well_mask[i, int(well_locs[i])] = 1.0
        self.fields = lnk.unsqueeze(1)

        # 坐标归一化（无修改）
        def transform_coords_1d(coords):
            cell_size = 1.0 / 64.0
            return coords * cell_size + cell_size / 2.0

        obs_coords_raw = data['obs_coords_1d']
        self.coords_raw = obs_coords_raw
        coords_norm = transform_coords_1d(torch.from_numpy(obs_coords_raw).float())
        self.coords = coords_norm.unsqueeze(-1)

        # ===================== 2. 水头值处理（核心优化） =====================
        h_raw = data['obs_values']
        self.h_raw = h_raw

        # 预加载归一化参数（测试/预测阶段）
        if norm_params_path and os.path.exists(norm_params_path):
            print(f"加载预存的归一化参数：{norm_params_path}")
            norm_params = np.load(norm_params_path)
            self.q_low = norm_params['q_low']
            self.q_high = norm_params['q_high']
            self.h_clipped_min = norm_params['h_clipped_min']
            self.h_clipped_max = norm_params['h_clipped_max']
            # 直接用预加载的参数截断+归一化
            h_clipped = np.clip(h_raw, self.q_low, self.q_high)
        else:
            # 训练阶段：计算分位数和截断参数
            h_flat = h_raw.flatten()
            self.q_low = np.percentile(h_flat, (1 - quantile) * 100 / 2)
            self.q_high = np.percentile(h_flat, 100 - (1 - quantile) * 100 / 2)
            h_clipped = np.clip(h_raw, self.q_low, self.q_high)
            self.h_clipped_min = h_clipped.min()
            self.h_clipped_max = h_clipped.max()

            # 保存归一化参数到文件（训练阶段）
            if save_norm_params:
                save_path = norm_params_path or "e1_1d_norm_params.npz"
                np.savez(
                    save_path,
                    quantile=quantile,
                    q_low=self.q_low,
                    q_high=self.q_high,
                    h_clipped_min=self.h_clipped_min,
                    h_clipped_max=self.h_clipped_max
                )
                print(f"归一化参数已保存到：{save_path}")

        # 筛选有效样本（无修改）
        h_sample_mean = np.mean(h_raw, axis=1)
        valid_sample_mask = (h_sample_mean >= self.q_low) & (h_sample_mean <= self.q_high)
        self.valid_sample_ids = np.where(valid_sample_mask)[0]
        print(f"原始样本数：{N} → 有效样本数：{len(self.valid_sample_ids)}")

        # 归一化水头值
        h_norm = (torch.from_numpy(h_clipped).float() - self.h_clipped_min) / (self.h_clipped_max - self.h_clipped_min)

        # 筛选有效样本数据
        self.fields = self.fields[self.valid_sample_ids]
        self.coords = self.coords[self.valid_sample_ids]
        self.heads = h_norm[self.valid_sample_ids]
        self.well_mask = well_mask[self.valid_sample_ids]

        # 输出关键信息
        print(f"坐标归一化范围：[{self.coords.min():.6f}, {self.coords.max():.6f}]")
        print(f"水头归一化范围：[{self.heads.min():.6f}, {self.heads.max():.6f}]")
        print("数据集初始化完成！")

    def inverse_normalize_head(self, h_norm):
        """反归一化（无修改）"""
        if isinstance(h_norm, torch.Tensor):
            h_real = h_norm * (self.h_clipped_max - self.h_clipped_min) + self.h_clipped_min
        else:
            h_real = h_norm * (self.h_clipped_max - self.h_clipped_min) + self.h_clipped_min
        return h_real

    def __len__(self):
        return len(self.valid_sample_ids)

    def __getitem__(self, idx):
        return self.fields[idx], self.coords[idx], self.heads[idx]


# ===================== 测试：训练/测试阶段分离 =====================
if __name__ == "__main__":
    # 1. 训练阶段：初始化数据集并保存归一化参数
    train_file = r"F:\0projects\deeponet_1d\data_1d\E1_1D_Final_Dataset_Pack_pos20.npz"
    train_dataset = DeepONetDataset_E1_1D(
        train_file,
        quantile=0.95,
        save_norm_params=True,
        norm_params_path="e1_1d_norm_params.npz"
    )

    # 2. 测试/预测阶段：加载预存的归一化参数（无需重新计算分位数）
    test_dataset = DeepONetDataset_E1_1D(
        train_file,  # 替换为测试集路径
        quantile=0.95,
        save_norm_params=False,
        norm_params_path="e1_1d_norm_params.npz"
    )

    # 验证反归一化
    _, _, h_norm = train_dataset[0]
    h_real = train_dataset.inverse_normalize_head(h_norm)
    print(f"\n归一化水头值：{h_norm[:5].numpy()}")
    print(f"反归一化后真实水头值：{h_real[:5].numpy()}")