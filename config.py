import os
from datetime import datetime


class Config:
    # --- 实验标识 ---
    EXP_NAME = "E1_1D"  # 一维E1实验
    MODEL_TYPE = "Vanilla"  # "Vanilla" 或 "Modified"

    # --- 路径设置 ---
    DATA_PATH = rf'F:\0projects\deeponet_1d\data_1d\{EXP_NAME}_Final_Dataset_Pack_pos20.npz'
    # DATA_PATH = rf'F:\0projects\deeponet_1d\data_1d\{EXP_NAME}_Final_Dataset_Pack_rev20.npz'
    OUTPUT_BASE = "./outputs_1D"
    TIMESTAMP = datetime.now().strftime('%m%d_%H%M')
    SAVE_DIR = os.path.join(OUTPUT_BASE, f"{EXP_NAME}_{MODEL_TYPE}_{TIMESTAMP}")

    # --- 训练超参数 ---
    SEED = 42
    BATCH_SIZE = 128
    LR = 1e-4
    EPOCHS = 5000
    SPLIT = 0.8
    P_DIM = 150  # DeepONet 的基数 p