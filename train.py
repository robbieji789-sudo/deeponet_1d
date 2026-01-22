import os
import json
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from datetime import datetime

from config import Config
from models import Vanilla_CNN_DeepONet_1D, Modified_CNN_DeepONet_1D
from dataset import DeepONetDataset_E1_1D

import matplotlib
matplotlib.use("Agg")


def main():
    # 1. 初始化环境与文件夹
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 数据准备
    # 训练阶段保存归一化参数，路径指向输出目录
    norm_params_path = os.path.join(Config.SAVE_DIR, "norm_params.npz")
    dataset = DeepONetDataset_E1_1D(
        Config.DATA_PATH,
        quantile=0.95,
        save_norm_params=True,
        norm_params_path=norm_params_path  # 归一化参数保存到实验目录
    )

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_idx = int(Config.SPLIT * len(indices))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=Config.BATCH_SIZE)

    # 3. 模型实例化
    if Config.MODEL_TYPE == "Vanilla":
        model = Vanilla_CNN_DeepONet_1D(p=Config.P_DIM).to(device)
    else:
        model = Modified_CNN_DeepONet_1D(p=Config.P_DIM).to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    criterion = nn.MSELoss()

    # 4. 训练循环
    best_loss = float('inf')
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    print(f"--- 启动一维实验: {Config.EXP_NAME} | 模型架构: {Config.MODEL_TYPE} ---")
    print(f"模型基数 P_DIM: {Config.P_DIM} | 归一化范围 h_min: {dataset.h_clipped_min:.6f}, h_max: {dataset.h_clipped_max:.6f}")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_mse = 0
        for f, q, y in train_loader:
            # 适配Modified模型输入（拼接lnK场和井掩码）
            if Config.MODEL_TYPE == "Modified":
                # 获取对应idx的井掩码并拼接 (B,1,64) + (B,1,64) -> (B,2,64)
                batch_indices = train_loader.dataset.indices[train_loader._iterator._idxs[:f.shape[0]]]
                well_mask = dataset.well_mask[batch_indices].unsqueeze(1)
                f = torch.cat([f, well_mask], dim=1)

            f, q, y = f.to(device), q.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(f, q)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_mse += loss.item() * f.size(0)

        model.eval()
        val_mse = 0
        with torch.no_grad():
            for f, q, y in test_loader:
                # 适配Modified模型输入
                if Config.MODEL_TYPE == "Modified":
                    batch_indices = test_loader.dataset.indices[test_loader._iterator._idxs[:f.shape[0]]]
                    well_mask = dataset.well_mask[batch_indices].unsqueeze(1)
                    f = torch.cat([f, well_mask], dim=1)

                f, q, y = f.to(device), q.to(device), y.to(device)
                val_mse += criterion(model(f, q), y).item() * f.size(0)

        avg_t, avg_v = train_mse / len(train_idx), val_mse / len(test_idx)

        # 记录数据
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_t)
        history["val_loss"].append(avg_v)

        scheduler.step(avg_v)

        # 保存最优模型
        if avg_v < best_loss:
            best_loss = avg_v
            torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "model_best.pth"))
            # 保存最优模型时同步记录当前最优val loss
            best_info = {"best_val_loss": best_loss, "best_epoch": epoch + 1}
            with open(os.path.join(Config.SAVE_DIR, "best_info.json"), "w") as f:
                json.dump(best_info, f, indent=4)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1:04d} | Train MSE: {avg_t:.4e} | Val MSE: {avg_v:.4e}")

    # 5. 保存 Loss 数据为 CSV
    history_csv = os.path.join(Config.SAVE_DIR, "training_history.csv")
    with open(history_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i in range(len(history["epoch"])):
            writer.writerow([history["epoch"][i], history["train_loss"][i], history["val_loss"][i]])
    print(f"Loss 数据已保存至: {history_csv}")

    # 6. 保存可视化图表
    plt.figure(figsize=(10, 6))
    plt.semilogy(history["epoch"], history["train_loss"], label='Train Loss')
    plt.semilogy(history["epoch"], history["val_loss"], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'1D Training History - {Config.EXP_NAME} ({Config.MODEL_TYPE})')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig(os.path.join(Config.SAVE_DIR, "loss_curve.png"))
    plt.close()

    # 7. 保存实验元数据供 predict.py 使用
    config_record = {
        "exp_name": Config.EXP_NAME,
        "model_type": Config.MODEL_TYPE,
        "p_dim": Config.P_DIM,  # 保存模型核心超参数P_DIM
        "norm": {
            "h_min": float(dataset.h_clipped_min),  # 浮点型保证json序列化
            "h_max": float(dataset.h_clipped_max),
            "q_low": float(dataset.q_low),
            "q_high": float(dataset.q_high)
        },
        "training": {
            "seed": Config.SEED,
            "batch_size": Config.BATCH_SIZE,
            "lr": Config.LR,
            "epochs": Config.EPOCHS,
            "split": Config.SPLIT,
            "best_val_loss": float(best_loss)
        },
        "indices": {
            "train": [int(idx) for idx in train_idx],  # 确保为int类型
            "test": [int(idx) for idx in test_idx]
        }
    }
    with open(os.path.join(Config.SAVE_DIR, "config.json"), "w") as f:
        json.dump(config_record, f, indent=4)

    print(f"一维实验完成！所有文件已保存至: {Config.SAVE_DIR}")
    print(f"关键参数记录：P_DIM={Config.P_DIM}, 水头归一化范围 [{dataset.h_clipped_min:.6f}, {dataset.h_clipped_max:.6f}]")


if __name__ == "__main__":
    main()