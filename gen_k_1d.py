from KLE import *
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use("Agg")
# 忽略所有警告
warnings.filterwarnings("ignore")


def generate_k_field_1d(eta, L_x, var, n_eigen, weight, seed=100):
    """生成一维渗透率场"""
    np.random.seed(seed)

    # 1. 计算一维特征值和特征根
    # 调用 KLE.py 中的一维解法
    lamda_x, w_x0, cum_lamda = eigen_value_solution(eta, L_x, var, n_eigen)

    # 2. 根据权重截断特征值 (一维无需 sort_lamda)
    n = n_eigen
    for k in range(len(cum_lamda)):
        # 贡献率计算：累计特征值和 / (长度 * 方差)
        if cum_lamda[k] / (L_x * var) >= weight:
            n = k + 1
            break

    # 截断特征值和频率根
    lamda = lamda_x[:n]
    w_x = w_x0[:n]

    # 3. 生成标准正态分布随机变量
    kesi = np.random.randn(n)

    # 4. 创建一维网格
    x = np.linspace(0, L_x, 64)

    # 5. 计算一维特征函数
    # fx 形状为 (n, len(x))
    fx = eigen_func2(n, w_x, eta, L_x, x)

    # 6. 叠加生成对数渗透率场 logK(x)
    logk_field = np.zeros(len(x))
    for i in range(n):
        logk_field += np.sqrt(lamda[i]) * kesi[i] * fx[i, :]

    # 7. 转换为渗透率场 K
    k_field = np.exp(logk_field)

    return k_field, x, n


if __name__ == '__main__':
    # 参数设置
    n_k = 1000  # 总生成组数
    Lx = 640.0  # x方向长度
    eta = 640.0 * 0.2  # 相关长度
    var = 1.0  # 方差
    n_eigen = 100  # 最大特征值个数
    weight = 0.95  # 能量权重 (1D建议设置较高)

    k_fields = []
    n_values = []

    # 循环生成数据
    for seed in range(n_k):
        k_field, x, n = generate_k_field_1d(eta, Lx, var, n_eigen, weight, seed=seed)
        k_fields.append(k_field)
        n_values.append(n)

        if (seed + 1) % 20 == 0:
            print(f"Generated field {seed + 1}/{n_k}: n={n}, K range={k_field.min():.4f} to {k_field.max():.4f}")

    # 转换为numpy数组 (100, 64)
    k_fields_array = np.array(k_fields)

    # 创建保存目录
    data_dir = 'data_1d'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 保存 .npy 数据
    save_path = os.path.join(data_dir, 'data_K_1d.npy')
    np.save(save_path, k_fields_array)
    print(f"\nSaved 1D K fields to {save_path}")

    # --- 画图检查部分 ---
    plt.figure(figsize=(10, 6))

    # 画出前 5 个场进行检查
    for i in range(5):
        plt.plot(x, np.log(k_fields[i]), label=f'Seed {i} (n={n_values[i]})')

    plt.title('Check: First 5 1D Log-Permeability Fields (ln K)')
    plt.xlabel('Distance X (m)')
    plt.ylabel('ln K')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存图片
    img_save_path = os.path.join(data_dir, 'check_plot_1d.png')
    plt.savefig(img_save_path, dpi=300)
    print(f"Check plot saved to {img_save_path}")

    plt.show()

    print(f"Average eigenvalues used: {np.mean(n_values):.2f}")