import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib

# 从自定义模块导入一维模型和数据集类
from models import Vanilla_CNN_DeepONet_1D, Modified_CNN_DeepONet_1D
from dataset import DeepONetDataset_E1_1D

# 设置matplotlib后端为Agg，适配无GUI环境（服务器/命令行）
matplotlib.use("Agg")


def evaluate_and_plot_1d(folder_path, plot_sample_indices=[0, 1, 2]):
    """
    一维DeepONet测试集评估与绘图函数：
    核心功能：
    1. 加载一维E1实验的训练模型，预测1D水头值
    2. 绘制1x2布局的科研级对比图：
       - 子图1：lnK场(1D) + 观测点水头真值(蓝色)vs预测值(红色)
       - 子图2：测试集所有观测点的真值vs预测值散点图（含MSE/R²）
    3. 完全适配train.py输出的config.json/norm_params.npz

    参数：
        folder_path (str): 模型输出文件夹路径（如E1_1D_Vanilla_0122_1558）
        plot_sample_indices (list): 要绘制的测试集样本索引列表
    """
    # ========== 1. 设备配置与文件检查 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 检查关键文件是否存在
    config_path = os.path.join(folder_path, "config.json")
    model_path = os.path.join(folder_path, "model_best.pth")
    norm_params_path = os.path.join(folder_path, "norm_params.npz")

    for file_path in [config_path, model_path, norm_params_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"关键文件缺失: {file_path}")

    # ========== 2. 加载配置与归一化参数 ==========
    # 加载实验配置
    with open(config_path, "r") as f:
        config = json.load(f)

    exp_name = config["exp_name"]
    model_type = config["model_type"]
    p_dim = config["p_dim"]
    test_idx = config["indices"]["test"]
    h_min = config["norm"]["h_min"]
    h_max = config["norm"]["h_max"]

    # 加载归一化参数（备用，优先使用config中的值）
    norm_params = np.load(norm_params_path)
    print(f"加载归一化参数：q_low={norm_params['q_low']:.4f}, q_high={norm_params['q_high']:.4f}")
    print(f"水头归一化范围：h_min={h_min:.4f}, h_max={h_max:.4f}")

    # ========== 3. 加载数据集（测试阶段） ==========
    # 从config中读取数据路径
    data_path = config.get("training", {}).get(
        "data_path") or r"F:\0projects\deeponet_1d\data_1d\E1_1D_Final_Dataset_Pack_pos20.npz"

    # 测试阶段：加载预存的归一化参数，不重新计算
    dataset = DeepONetDataset_E1_1D(
        file_path=data_path,
        quantile=0.95,
        save_norm_params=False,
        norm_params_path=norm_params_path
    )
    # 构建测试集Subset
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    print(f"测试集样本数: {len(test_dataset)}")

    # ========== 4. 加载模型并设置为评估模式 ==========
    if model_type == "Vanilla":
        model = Vanilla_CNN_DeepONet_1D(p=p_dim).to(device)
    else:  # Modified
        model = Modified_CNN_DeepONet_1D(p=p_dim).to(device)

    # 加载最优模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"成功加载{model_type}模型，参数维度p={p_dim}")

    # ========== 5. 全测试集预测与指标计算 ==========
    all_true = []
    all_pred = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            # 获取单个样本数据
            f, q, y_true = test_dataset[idx]
            f = f.unsqueeze(0).to(device)  # (1, 1, 64)
            q = q.unsqueeze(0).to(device)  # (1, M, 1)

            # Modified模型需要拼接井掩码
            if model_type == "Modified":
                # 获取原始样本ID -> 井掩码
                original_idx = test_idx[idx]
                well_mask = dataset.well_mask[original_idx].unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64)
                f = torch.cat([f, well_mask], dim=1)  # (1,2,64)

            # 预测并反归一化
            y_pred_norm = model(f, q)
            y_pred = dataset.inverse_normalize_head(y_pred_norm.cpu()).numpy().flatten()
            y_true = dataset.inverse_normalize_head(y_true.cpu()).numpy().flatten()

            all_true.extend(y_true)
            all_pred.extend(y_pred)

    # 计算全测试集评估指标
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    mse_total = mean_squared_error(all_true, all_pred)
    r2_total = r2_score(all_true, all_pred)
    print(f"\n全测试集评估结果：")
    print(f"MSE: {mse_total:.4e} | R²: {r2_total:.4f}")

    # ========== 6. 绘制指定样本的可视化图 ==========
    for plot_idx in plot_sample_indices:
        if plot_idx >= len(test_dataset):
            print(f"跳过超出范围的索引: {plot_idx} (测试集仅{len(test_dataset)}个样本)")
            continue

        # 获取当前样本数据
        original_sample_id = test_idx[plot_idx]
        f, q, y_true_norm = test_dataset[plot_idx]
        f = f.unsqueeze(0).to(device)
        q = q.unsqueeze(0).to(device)

        # Modified模型拼接井掩码
        if model_type == "Modified":
            well_mask = dataset.well_mask[original_sample_id].unsqueeze(0).unsqueeze(0).to(device)
            f = torch.cat([f, well_mask], dim=1)

        # 模型预测
        with torch.no_grad():
            y_pred_norm = model(f, q)

        # 反归一化得到真实水头值
        y_true = dataset.inverse_normalize_head(y_true_norm).numpy()
        y_pred = dataset.inverse_normalize_head(y_pred_norm.cpu()).numpy().flatten()

        # ========== 7. 1D可视化绘图 ==========
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 子图1：1D lnK场 + 水头值对比
        ax1 = axes[0]
        # 绘制lnK场（1D，y轴为lnK值，x轴为位置）
        lnk_field = dataset.fields[original_sample_id].squeeze().numpy()  # (64,)
        x_pos = np.arange(64) * (1.0 / 64) + (1.0 / 64) / 2  # 归一化坐标
        ax1.plot(x_pos, lnk_field, color='gray', linewidth=2, label='lnK Field')
        ax1.fill_between(x_pos, lnk_field, alpha=0.3, color='gray')

        # 绘制观测点水头真值和预测值
        obs_coords = dataset.coords[original_sample_id].squeeze().numpy()  # (M,)
        ax1.scatter(obs_coords, y_true, color='blue', s=50, label='True Head', zorder=3)
        ax1.scatter(obs_coords, y_pred, color='red', marker='x', s=50, label='Pred Head', zorder=4)

        # 标注抽水井位置
        # well_loc = dataset.coords_raw[original_sample_id][int(dataset.well_loc_1d[original_sample_id])]
        # ax1.axvline(x=well_loc * (1.0 / 64) + (1.0 / 64) / 2, color='black', linestyle='--', label='Pumping Well')

        ax1.set_title(f"1D lnK Field & Head Values (Sample {original_sample_id})", fontsize=12)
        ax1.set_xlabel("Normalized Position", fontsize=10)
        ax1.set_ylabel("Value", fontsize=10)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 子图2：当前样本真值vs预测值散点图 + 全测试集指标
        ax2 = axes[1]
        # 当前样本散点
        ax2.scatter(y_true, y_pred, color='gold', edgecolors='black', s=50, alpha=0.8,
                    label=f'Sample {original_sample_id}')
        # 1:1参考线
        lims = [
            min(y_true.min(), y_pred.min()) - 0.1,
            max(y_true.max(), y_pred.max()) + 0.1
        ]
        ax2.plot(lims, lims, 'r--', linewidth=2, label='1:1 Line')

        # 计算当前样本指标
        mse_sample = mean_squared_error(y_true, y_pred)
        r2_sample = r2_score(y_true, y_pred)

        # 标注指标
        ax2.text(0.05, 0.95,
                 f"Sample MSE: {mse_sample:.4e}\nSample R²: {r2_sample:.4f}\nTotal Test MSE: {mse_total:.4e}\nTotal Test R²: {r2_total:.4f}",
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax2.set_title("True vs Predicted Head Values", fontsize=12)
        ax2.set_xlabel("True Head (m)", fontsize=10)
        ax2.set_ylabel("Predicted Head (m)", fontsize=10)
        ax2.legend()
        ax2.grid(alpha=0.3)

        # ========== 8. 保存图片 ==========
        plt.tight_layout()
        save_path = os.path.join(folder_path, f"1D_test_sample_{plot_idx}_id_{original_sample_id}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ 样本{plot_idx}可视化图已保存: {save_path}")

    # ========== 9. 保存全测试集评估结果 ==========
    eval_result = {
        "total_test_samples": len(test_dataset),
        "total_observation_points": len(all_true),
        "mse": float(mse_total),
        "r2": float(r2_total),
        "h_min": float(h_min),
        "h_max": float(h_max),
        "model_type": model_type,
        "p_dim": p_dim
    }
    with open(os.path.join(folder_path, "test_evaluation.json"), "w") as f:
        json.dump(eval_result, f, indent=4)
    print(f"\n📊 全测试集评估结果已保存至: {os.path.join(folder_path, 'test_evaluation.json')}")


if __name__ == "__main__":
    # 替换为你的模型输出文件夹路径
    TARGET_DIR = r"F:\0projects\deeponet_1d\outputs_1D\E1_1D_Vanilla_0122_1658"
    # 绘制测试集前3个样本
    evaluate_and_plot_1d(TARGET_DIR, plot_sample_indices=[0, 1, 2])