import numpy as np
import matplotlib.pyplot as plt
import flopy
import warnings
import os
import matplotlib
matplotlib.use("Agg")
# 忽略所有警告
warnings.filterwarnings("ignore")


def run_1d_simulation(data_K, Lx, ncol, well_col, q, model_ws, modelname_suffix):
    """
    通用函数：运行一维抽水模拟
    well_col: 井的列索引 (1-based)
    """
    nrow = 1
    delr = Lx / ncol
    delc = 1.0
    n_samples = data_K.shape[0]
    data_u = np.zeros_like(data_K)

    if not os.path.exists(model_ws):
        os.makedirs(model_ws)

    for i in range(n_samples):
        print(i)
        # 获取当前渗透系数并转为 MODFLOW 要求的形状 (nlay, nrow, ncol)
        current_k = data_K[i, :].reshape((1, 1, ncol))

        sim = flopy.mf6.MFSimulation(sim_name=f"sim_{modelname_suffix}_{i}", exe_name="mf6", sim_ws=model_ws)
        tdis = flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
        ims = flopy.mf6.ModflowIms(sim, complexity="SIMPLE", outer_dvclose=1e-6)
        gwf = flopy.mf6.ModflowGwf(sim, modelname=f"gwf_{i}", save_flows=True)

        dis = flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=nrow, ncol=ncol, delr=delr, delc=delc, top=0.0, botm=-1.0)
        ic = flopy.mf6.ModflowGwfic(gwf, strt=0.0)
        npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=current_k)

        # 左右边界水头为 0
        chd_spd = [[(0, 0, 0), 0.0], [(0, 0, ncol - 1), 0.0]]
        flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)

        # 单井抽水
        wel_spd = [[(0, 0, well_col - 1), q]]
        flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd)

        oc = flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f"head_{i}.hds", saverecord=[("HEAD", "ALL")])

        sim.write_simulation(silent=True)
        success, _ = sim.run_simulation(silent=True)

        if success:
            head_file = os.path.join(model_ws, f"head_{i}.hds")
            hds = flopy.utils.binaryfile.HeadFile(head_file)
            data_u[i, :] = hds.get_data(totim=1.0)[0, 0, :]

    return data_u


if __name__ == '__main__':
    # 基础参数
    Lx = 640.0
    ncol = 64
    q = -0.1
    data_K_path = "./data_1d/data_K_1d.npy"
    data_K = np.load(data_K_path)
    save_dir = os.path.dirname(data_K_path)

    # --- 情况 1: 正数第 20 (well_col = 20) ---
    print("开始生成数据集：正数第 20 个单元格抽水...")
    u_pos20 = run_1d_simulation(data_K, Lx, ncol, 20, q, "./mf6_pos20", "pos20")
    np.save(os.path.join(save_dir, "data_u_pos20.npy"), u_pos20)

    # --- 情况 2: 倒数第 20 (well_col = 64 - 20 + 1 = 45) ---
    print("开始生成数据集：倒数第 20 个单元格抽水...")
    u_rev20 = run_1d_simulation(data_K, Lx, ncol, 45, q, "./mf6_rev20", "rev20")
    np.save(os.path.join(save_dir, "data_u_rev20.npy"), u_rev20)

    # --- 可视化对比 ---
    plt.figure(figsize=(10, 5))
    x = np.linspace(0, Lx, ncol)
    plt.plot(x, u_pos20[0, :], 'b-', label='Well at Col 20 (Sample 0)')
    plt.plot(x, u_rev20[0, :], 'r--', label='Well at Col 45 (Sample 0)')
    plt.title('Comparison of Head Response (Single Well at Different Locations)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Head (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "comparison_1d.png"))
    plt.show()

    print(f"任务完成！数据集已保存至 {save_dir}")