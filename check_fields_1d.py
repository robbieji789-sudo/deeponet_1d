import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

# 设置后端避免显示问题，支持无界面保存图片
matplotlib.use("Agg")

# ===================== 基础参数配置（与生成代码完全对齐） =====================
Lx = 640.0  # x方向长度 (m)
ncol = 64  # 网格列数
data_dir = "./data_1d"  # 数据存储目录
# 数据文件路径
k_data_path = os.path.join(data_dir, "data_K_1d.npy")
u_pos20_path = os.path.join(data_dir, "data_u_pos20.npy")
u_rev20_path = os.path.join(data_dir, "data_u_rev20.npy")


# ===================== 加载数据 =====================
def load_data():
    """加载渗透率场和水头场数据，检查文件是否存在"""
    required_files = [k_data_path, u_pos20_path, u_rev20_path]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"数据文件缺失：{file}\n请先运行gen_k_1d.py和gen_u_1d.py生成数据")

    # 加载数据
    data_K = np.load(k_data_path)
    data_lnk = np.log(data_K)  # 对数渗透率场
    u_pos20 = np.load(u_pos20_path)  # 20列抽水的水头场
    u_rev20 = np.load(u_rev20_path)  # 45列抽水的水头场

    # 生成x轴坐标
    x = np.linspace(0, Lx, ncol)
    return data_lnk, u_pos20, u_rev20, x


# ===================== 新增：最小水头分布分析函数 =====================
def analyze_min_head_distribution(u_pos20, u_rev20, save_path):
    """
    分析并可视化水头场最小值的分布特征：
    1. 计算每个样本的最小水头值
    2. 统计最小值的均值、中位数、标准差、分位数
    3. 绘制直方图+核密度曲线展示分布
    """
    print("\n=== 一维水头场最小值分布分析 ===")

    # 1. 计算每个样本的最小水头值
    u_pos20_min = np.min(u_pos20, axis=1)  # (1000,) 每个样本pos20的最小水头
    u_rev20_min = np.min(u_rev20, axis=1)  # (1000,) 每个样本rev20的最小水头

    # 2. 统计分析
    def print_stats(data, name):
        print(f"\n{name} 最小水头统计：")
        print(f"  均值: {np.mean(data):.4f} m")
        print(f"  中位数: {np.median(data):.4f} m")
        print(f"  标准差: {np.std(data):.4f} m")
        print(f"  最小值: {np.min(data):.4f} m")
        print(f"  最大值: {np.max(data):.4f} m")
        print(f"  25分位数: {np.percentile(data, 25):.4f} m")
        print(f"  75分位数: {np.percentile(data, 75):.4f} m")

    print_stats(u_pos20_min, "u_pos20 (20列抽水)")
    print_stats(u_rev20_min, "u_rev20 (45列抽水)")

    # 3. 可视化分布（直方图+核密度曲线）
    plt.figure(figsize=(12, 6))

    # 绘制直方图
    bins = 50  # 分箱数
    plt.hist(u_pos20_min, bins=bins, alpha=0.6, label='u_pos20 (Well at Col 20)', color='blue', density=True)
    plt.hist(u_rev20_min, bins=bins, alpha=0.6, label='u_rev20 (Well at Col 45)', color='red', density=True)

    # 绘制核密度曲线（更平滑展示分布）
    from scipy.stats import gaussian_kde
    kde_pos20 = gaussian_kde(u_pos20_min)
    kde_rev20 = gaussian_kde(u_rev20_min)
    x_range = np.linspace(min(np.min(u_pos20_min), np.min(u_rev20_min)),
                          max(np.max(u_pos20_min), np.max(u_rev20_min)), 200)
    plt.plot(x_range, kde_pos20(x_range), 'b-', linewidth=2, label='u_pos20 KDE')
    plt.plot(x_range, kde_rev20(x_range), 'r-', linewidth=2, label='u_rev20 KDE')

    # 添加统计标注
    plt.axvline(np.mean(u_pos20_min), color='blue', linestyle='--', alpha=0.8,
                label=f'u_pos20 均值: {np.mean(u_pos20_min):.4f} m')
    plt.axvline(np.mean(u_rev20_min), color='red', linestyle='--', alpha=0.8,
                label=f'u_rev20 均值: {np.mean(u_rev20_min):.4f} m')

    # 图表样式设置
    plt.title("Distribution of Minimum Head Values (1000 Samples, 64 Columns)", fontsize=14, fontweight='bold')
    plt.xlabel("Minimum Head (m)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n最小水头分布分析图已保存：{save_path}")

    # 4. 输出最小值位置分布（可选：分析最小水头出现在哪个列）
    print("\n=== 最小水头位置分布 ===")
    u_pos20_min_col = np.argmin(u_pos20, axis=1)  # 每个样本pos20最小水头的列索引
    u_rev20_min_col = np.argmin(u_rev20, axis=1)
    print(f"u_pos20 最小水头列索引统计：")
    print(f"  最常出现列: {np.bincount(u_pos20_min_col).argmax()} (出现次数: {np.bincount(u_pos20_min_col).max()})")
    print(f"  列索引均值: {np.mean(u_pos20_min_col):.2f}")
    print(f"u_rev20 最小水头列索引统计：")
    print(f"  最常出现列: {np.bincount(u_rev20_min_col).argmax()} (出现次数: {np.bincount(u_rev20_min_col).max()})")
    print(f"  列索引均值: {np.mean(u_rev20_min_col):.2f}")


# ===================== 可视化检查函数（修复原代码笔误） =====================
def plot_check_fields(data_lnk, u_pos20, u_rev20, x, save_path):
    """
    绘制3行1列子图：
    - 每行对应1个样本（前3个）
    - 左纵轴：水头场（u_pos20=蓝线，u_rev20=红线）
    - 右纵轴：对数渗透率场（lnK=绿线）
    """
    # 创建3行1列子图，共享x轴
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle("Check: Head Fields & Log-Permeability Fields (First 3 Samples)",
                 fontsize=14, fontweight='bold', y=0.98)

    # 遍历前3个样本
    for idx, ax1 in enumerate(axes):
        # 左纵轴：水头场
        ax1.plot(x, u_pos20[idx, :], 'b-', linewidth=2, label=f'Well at Col 20 (Head)')
        ax1.plot(x, u_rev20[idx, :], 'r--', linewidth=2, label=f'Well at Col 45 (Head)')
        ax1.set_ylabel("Head (m)", fontsize=11, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title(f"Sample {idx + 1}", fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)

        # 右纵轴：对数渗透率场（修复原代码笔误：ccolor→color，labelcolor→color）
        ax2 = ax1.twinx()
        ax2.plot(x, data_lnk[idx, :], 'g-', linewidth=2, label='ln K (Permeability)')
        ax2.set_ylabel("ln K", fontsize=11, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.legend(loc='upper right', fontsize=9)

    # 设置公共x轴标签
    axes[-1].set_xlabel("Distance (m)", fontsize=12)

    # 调整子图间距，避免重叠
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # 保存高分辨率图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"检查图已保存：{save_path}")


# ===================== 主函数（新增调用最小水头分析） =====================
if __name__ == '__main__':
    # 1. 加载数据
    try:
        data_lnk, u_pos20, u_rev20, x = load_data()
    except FileNotFoundError as e:
        print(f"错误：{e}")
        exit(1)

    # 2. 确保保存目录存在
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 3. 生成并保存检查图（合并后的check_u+lnk图）
    plot_check_fields(data_lnk, u_pos20, u_rev20, x, os.path.join(data_dir, "check_u_lnk.png"))

    # 4. 新增：分析并可视化最小水头分布
    analyze_min_head_distribution(u_pos20, u_rev20, os.path.join(data_dir, "min_head_distribution.png"))

    # 5. 输出数据基本信息（辅助校验）
    print("\n=== 数据基本信息 ===")
    print(f"对数渗透率场 (ln K) 形状: {data_lnk.shape} (样本数 × 网格数)")
    print(f"水头场 (u_pos20) 形状: {u_pos20.shape}")
    print(f"水头场 (u_rev20) 形状: {u_rev20.shape}")
    print(f"前3个样本 ln K 范围: {data_lnk[:3].min():.4f} ~ {data_lnk[:3].max():.4f}")
    print(f"前3个样本 u_pos20 范围: {u_pos20[:3].min():.4f} ~ {u_pos20[:3].max():.4f}")
    print(f"前3个样本 u_rev20 范围: {u_rev20[:3].min():.4f} ~ {u_rev20[:3].max():.4f}")
    print("\n所有分析完成！")