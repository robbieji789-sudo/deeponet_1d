import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 适配无显示器环境
import matplotlib.pyplot as plt
import warnings

# 忽略无关警告
warnings.filterwarnings("ignore")

# ==========================================
# 1. 基础配置 (核心修改：num_obs改为64)
# ==========================================
# 路径配置 (修改为你的一维数据路径)
base_dir = r'F:\0projects\deeponet_1d\data_1d'
output_dir = base_dir  # 最终数据集保存目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 一维数据文件路径
k_data_path = os.path.join(base_dir, "data_K_1d.npy")
u_pos20_path = os.path.join(base_dir, "data_u_pos20.npy")
u_rev20_path = os.path.join(base_dir, "data_u_rev20.npy")  # 可选：如需rev20则启用

# 模型物理参数 (与一维生成代码对齐)
num_samples = 1000  # 一维数据共1000个样本
ncol = 64  # 一维网格列数
Lx = 640.0  # x方向长度 (m)
delr = Lx / ncol  # 网格步长
q_pumping = -1.0  # 抽水井流量 (与gen_u_1d.py一致)
well_col_pos20 = 20  # 20列抽水 (1-based)
well_col_rev20 = 45  # 45列抽水 (1-based)
num_obs = 64  # 核心修改：观测点数量改为64（与单元格数量一致）
dim_1d = ncol  # 一维维度：64

# ==========================================
# 2. 加载一维原始数据
# ==========================================
def load_1d_raw_data():
    print("--- 加载一维原始数据 ---")
    # 加载渗透率场 K -> 转换为对数渗透率 lnK
    data_K = np.load(k_data_path)
    lnK_data = np.log(data_K)  # (1000, 64)
    print(f"lnK 数据形状: {lnK_data.shape}")

    # 加载水头场 (选择pos20或rev20，二选一或都保留)
    # data_u = np.load(u_pos20_path)  # 20列抽水的水头场 (1000, 64)
    data_u = np.load(u_rev20_path)  # 如需rev20则注释上一行，启用此行
    print(f"水头场数据形状: {data_u.shape}")

    # 校验数据维度
    assert lnK_data.shape == (num_samples, ncol), f"lnK数据维度错误，应为({num_samples}, {ncol})"
    assert data_u.shape == (num_samples, ncol), f"水头数据维度错误，应为({num_samples}, {ncol})"

    return lnK_data, data_u

# 加载数据
lnK_data, data_u_full = load_1d_raw_data()

# ==========================================
# 3. 生成抽水井/观测井坐标 (核心修改：观测点覆盖所有64个单元格)
# ==========================================
def generate_1d_well_obs_coords(n_samples, well_col, num_obs, ncol):
    print("--- 生成一维抽水井/观测井坐标 ---")
    # 1. 抽水井坐标 (一维：列索引转一维索引，0-based)
    well_col_0based = well_col - 1  # 1-based转0-based
    well_1d = np.full((n_samples,), well_col_0based, dtype=int)  # (1000,) 所有样本井位固定

    # 2. 核心修改：观测井坐标改为所有64个单元格 (0~63)，不再随机选取
    # 所有样本的观测点都是0~63，无需随机
    obs_1d = np.tile(np.arange(ncol), (n_samples, 1))  # (1000, 64) 每个样本都是0-63

    # 3. 提取观测水头值：直接取完整水头场（因为观测点是所有单元格）
    obs_values = data_u_full  # (1000, 64) 无需循环提取，直接复用完整水头场

    return well_1d, obs_1d, obs_values

# 生成坐标和观测值 (使用pos20井位，如需rev20则替换well_col_rev20)
well_1d, obs_1d, obs_values = generate_1d_well_obs_coords(
    num_samples, well_col_pos20, num_obs, ncol
)

# ==========================================
# 4. 数据归一化 (可选，根据需求调整)
# ==========================================
print("--- 数据归一化处理 ---")
# lnK归一化 (0-1区间，可根据实际分布调整)
lnk_min = lnK_data.min()
lnk_max = lnK_data.max()
lnk_1d = (lnK_data - lnk_min) / (lnk_max - lnk_min)  # (1000, 64)

# 完整水头场无需归一化，保持原始值
u_full_1d = data_u_full  # (1000, 64)

# ==========================================
# 5. 可视化检查 (验证一维数据和坐标)
# ==========================================
def save_1d_preview(ids, lnk_1d, heads_1d, well_1d, obs_1d, save_name):
    print("--- 生成一维数据预览图 ---")
    x = np.linspace(0, Lx, ncol)  # x轴坐标
    fig, axes = plt.subplots(1, len(ids), figsize=(20, 4))
    for idx, s_id in enumerate(ids):
        ax = axes[idx]
        # 绘制lnK场
        ax.plot(x, lnk_1d[s_id], 'g-', linewidth=2, label='lnK (normalized)')
        # 绘制水头场
        ax.plot(x, heads_1d[s_id], 'b-', linewidth=2, label='Head')
        # 标记抽水井
        well_x = x[well_1d[s_id]]
        ax.scatter(well_x, heads_1d[s_id, well_1d[s_id]],
                   color='red', marker='x', s=80, label='Well')
        # 标记观测井（所有单元格，用浅灰色小点）
        obs_x = x[obs_1d[s_id]]
        ax.scatter(obs_x, heads_1d[s_id, obs_1d[s_id]],
                   color='gray', s=5, alpha=0.8, label='Obs Points (All Cells)')
        ax.set_title(f"Sample {s_id} (1D Check)")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"预览图已保存: {save_path}")

# 生成前5个样本的预览图
save_1d_preview([0, 1, 2, 3, 4], lnk_1d, u_full_1d, well_1d, obs_1d, "1d_data_check.png")

# ==========================================
# 6. 保存1D格式最终数据集
# ==========================================
# final_1d_path = os.path.join(output_dir, 'E1_1D_Final_Dataset_Pack_pos20.npz')
final_1d_path = os.path.join(output_dir, 'E1_1D_Final_Dataset_Pack_rev20.npz')
np.savez_compressed(
    final_1d_path,
    lnk_field_1d=lnk_1d,          # 归一化lnK场 (1000, 64)
    u_full_1d=u_full_1d,          # 完整水头场 (1000, 64)
    well_loc_1d=well_1d,          # 抽水井一维索引 (1000,)
    obs_values=obs_values,        # 观测水头值 (1000, 64) （修改后维度）
    obs_coords_1d=obs_1d,         # 观测井一维索引 (1000, 64) （修改后维度）
    lnk_min=lnk_min,              # 保存归一化参数，便于后续反归一化
    lnk_max=lnk_max,
    x_coords=np.linspace(0, Lx, ncol)  # 保存x轴坐标
)

# ==========================================
# 7. 输出数据集信息
# ==========================================
print(f"\n--- 1D数据集生成完成！---")
print(f"最终文件路径: {final_1d_path}")
print(f"\n数据集维度说明：")
print(f"- lnk_field_1d (归一化lnK场): {lnk_1d.shape}")
print(f"- u_full_1d (完整水头场): {u_full_1d.shape}")
print(f"- well_loc_1d (抽水井一维索引): {well_1d.shape}")
print(f"- obs_values (观测水头值): {obs_values.shape}")  # 输出修改后的维度
print(f"- obs_coords_1d (观测井一维索引): {obs_1d.shape}")  # 输出修改后的维度
print(f"\nlnK归一化参数：min={lnk_min:.4f}, max={lnk_max:.4f}")
print(f"抽水井位置 (0-based列索引): {well_1d[0]} (第{well_col_pos20}列)")
print(f"观测点数量：{num_obs} (覆盖所有{ncol}个单元格)")