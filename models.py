import torch
import torch.nn as nn


class Vanilla_CNN_DeepONet_1D(nn.Module):
    """一维Vanilla版DeepONet：1D卷积Branch + 全连接Trunk"""

    def __init__(self, p=150):
        super().__init__()
        # 输入 1x64 -> 6次池化 -> 1x1
        self.branch_conv = nn.Sequential(
            # 1D卷积替代2D卷积，适配一维输入
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2),  # 64->32
            # nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(),

            nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2),  # 32->16
            # nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(),

            nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2),  # 16->8
            # nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(),

            nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2),  # 8->4
            # nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(),

            nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2),  # 4->2
            # nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(),

            nn.Conv1d(16, 64, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2),  # 2->1
            nn.Flatten(),
            nn.Linear(1 * 64, 256), nn.Tanh(),
            nn.Linear(256, 512), nn.Tanh(),
            nn.Linear(512, p), nn.Tanh()
        )
        # Trunk分支：适配一维坐标输入(输入维度=1)
        self.trunk = nn.Sequential(
            nn.Linear(1, p), nn.ReLU(),
            nn.Linear(p, p), nn.ReLU(),
            nn.Linear(p, p), nn.ReLU(),
            nn.Linear(p, p), nn.ReLU(),
            nn.Linear(p, p), nn.ReLU()
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_field, x_query):
        """
        前向传播：
        :param x_field: (B, 1, 64) 一维场输入
        :param x_query: (B, M, 1) 一维查询坐标
        :return: (B, M) 预测水头值
        """
        B = self.branch_conv(x_field)  # (B, p)
        T = self.trunk(x_query)  # (B, M, p)
        return torch.einsum("bp,blp->bl", B, T) + self.bias


class Modified_CNN_DeepONet_1D(nn.Module):
    """一维Modified版DeepONet：带分支交互的1D DeepONet"""

    def __init__(self, p=150):
        super().__init__()
        # Trunk 隐藏层维度
        self.t_dims = [4096, 1024, 256, 256, 64, 64]
        # 一维分辨率演变: 64->32, 32->16, 16->8, 8->4, 4->2, 2->1
        self.res = [32, 16, 8, 4, 2, 1]

        # --- Branch 1D卷积层 ---
        # 前5层：输入 (16+16=32) -> 输出 16（第1层输入为2）
        self.b_conv1 = nn.Sequential(nn.Conv1d(2, 16, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2))
        self.b_conv2 = nn.Sequential(nn.Conv1d(32, 16, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2))
        self.b_conv3 = nn.Sequential(nn.Conv1d(32, 16, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2))
        self.b_conv4 = nn.Sequential(nn.Conv1d(32, 16, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2))
        self.b_conv5 = nn.Sequential(nn.Conv1d(32, 16, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2))

        # 第6层：输入 (16+16=32) -> 输出 64
        self.b_conv6 = nn.Sequential(nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2))

        # --- Trunk 线性层 (适配一维输入) ---
        self.t_fcs = nn.ModuleList([
            nn.Linear(1, self.t_dims[0]),
            nn.Linear(self.t_dims[0], self.t_dims[1]),
            nn.Linear(self.t_dims[1], self.t_dims[2]),
            nn.Linear(self.t_dims[2], self.t_dims[3]),
            nn.Linear(self.t_dims[3], self.t_dims[4]),
            nn.Linear(self.t_dims[4], self.t_dims[5])
        ])

        # --- 交互层 (Branch -> Trunk 系数) ---
        self.inter_fcs = nn.ModuleList([
            nn.Sequential(nn.Flatten(), nn.Linear(16 * self.res[i], self.t_dims[i]), nn.Sigmoid())
            if i < 5 else nn.Sequential(nn.Flatten(), nn.Linear(64 * 1, self.t_dims[i]), nn.Sigmoid())
            for i in range(6)
        ])

        # --- 反馈层 (Trunk -> Branch) ---
        # 前5层反馈 16 通道，第6层反馈 64 通道
        self.feedback_fcs = nn.ModuleList([
            nn.Linear(self.t_dims[i], 16) if i < 5 else nn.Linear(self.t_dims[i], 64)
            for i in range(6)
        ])

        # --- 最终输出层 ---
        self.branch_final = nn.Sequential(
            nn.Flatten(),
            # 此时输入是 64(卷积输出) + 64(反馈输出) = 128
            nn.Linear(128 * 1, 256), nn.Tanh(),
            nn.Linear(256, 512), nn.Tanh(),
            nn.Linear(512, p)
        )
        self.t_out = nn.Linear(self.t_dims[-1], p)
        self.act = nn.ReLU()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_field, x_query):
        """
        前向传播：
        :param x_field: (B, 2, 64) 一维场输入（lnK+井掩码）
        :param x_query: (B, M, 1) 一维查询坐标
        :return: (B, M) 预测水头值
        """
        b = x_field
        t = x_query
        b_conv_list = [self.b_conv1, self.b_conv2, self.b_conv3, self.b_conv4, self.b_conv5, self.b_conv6]

        for i in range(6):
            # 1. Branch 1D卷积推进
            b = b_conv_list[i](b)

            # 2. Trunk 推进并注入 Branch 权重 (B -> T)
            coeff = self.inter_fcs[i](b).unsqueeze(1)
            t = self.act(self.t_fcs[i](t))
            t = t * coeff

            # 3. 反馈给 Branch (T -> B) - 每一层都执行
            # 获取当前层反馈所需的通道数 (16 或 64)
            current_feedback_channels = 16 if i < 5 else 64

            f_info = self.feedback_fcs[i](t.mean(dim=1))
            f_info = f_info.view(-1, current_feedback_channels, 1)
            f_info = f_info.expand(-1, -1, self.res[i])

            # 拼接
            b = torch.cat([b, f_info], dim=1)

        B = self.branch_final(b)
        T = self.t_out(t)
        return torch.einsum("bp,blp->bl", B, T) + self.bias


# --- 测试脚本 ---
if __name__ == "__main__":
    # 测试一维Modified模型
    model = Modified_CNN_DeepONet_1D(p=150)
    x_f = torch.randn(32, 2, 64)  # (B, C, L)
    x_q = torch.randn(32, 100, 1)  # (B, M, 1)
    output = model(x_f, x_q)
    print('***Modified_CNN_DeepONet_1D***')
    print(f"输入场形状: {x_f.shape}")
    print(f"输出结果形状: {output.shape}")  # 预期 [32, 100]

    # 测试一维Vanilla模型
    model = Vanilla_CNN_DeepONet_1D(p=150)
    x_f = torch.randn(32, 1, 64)
    x_q = torch.randn(32, 100, 1)
    output = model(x_f, x_q)
    print('***Vanilla_CNN_DeepONet_1D***')
    print(f"输入场形状: {x_f.shape}")
    print(f"输出结果形状: {output.shape}")  # 预期 [32, 100]