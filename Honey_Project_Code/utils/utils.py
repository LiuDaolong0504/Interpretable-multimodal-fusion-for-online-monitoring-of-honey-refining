from typing import Optional, Tuple, Union
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import seaborn as sns
from scipy.stats import pearsonr


# 设置全局字体为 "Times New Roman"
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 9         # 设置全局字体大小
plt.rcParams['axes.unicode_minus'] = False  # 设置支持负号显示
plt.rcParams['svg.fonttype'] = 'none'


def _ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def dataloader(
        data: np.ndarray,
        label: np.ndarray,
        test_size: float = 0.2,
        seed: int = 42,
        batch_size: int = 16
) -> Tuple[dict, dict, dict]:
    """自定义的DataLoader"""
    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    # 将数据转换为tensor
    features_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    labels_tensor = torch.tensor(label, dtype=torch.float32)
    # 首先划分出测试集 (20%) 和剩余的部分 (80%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_tensor, labels_tensor, test_size=test_size, random_state=seed)
    # 再将剩余部分划分为验证集 (20%) 和训练集 (60%)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=seed)

    print(f"训练集: {X_train.shape}, {y_train.shape}")
    print(f"验证集: {X_val.shape}, {y_val.shape}")
    print(f"测试集: {X_test.shape}, {y_test.shape}")

    # 1.7 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test,  y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model,
                dataloader: list,
                epochs: int,
                criterion,
                optimizer,
                bestmodel_path: str,
                scheduler,
                ifscheduler=True,
                device='cuda'):
    save_dir = './model_log/weight_ft'
    _ensure_dir(os.path.join(save_dir, bestmodel_path))

    train_losses = []
    val_losses = []

    best_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        # ----------------------
        # 在训练集上训练模型
        # ----------------------
        model.train()
        total_loss = 0.0

        for x, y in dataloader[0]:
            x, y = x.to(device), y.to(device)
            # x = x.unsqueeze(1)
            optimizer.zero_grad()  # 梯度清零
            output_x = model(x)  # 前向传播
            loss = criterion(output_x.view(-1), y.view(-1))  # 计算损失
            loss.backward()  # 后向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 记录损失

        if ifscheduler:
            scheduler.step()

        train_loss = total_loss / len(dataloader[0])
        train_losses.append(train_loss)

        # ----------------------
        # 在验证集上验证模型
        # ----------------------
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for x_val, y_val in dataloader[1]:
                x_val, y_val = x_val.to(device), y_val.to(device)
                # x_val = x_val.unsqueeze(1)
                output_val = model(x_val)
                val_loss = criterion(output_val.view(-1), y_val.view(-1))
                total_val_loss += val_loss.item()

        # 计算平均损失
        val_loss = total_val_loss / len(dataloader[1])
        val_losses.append(val_loss)

        # 保存验证集上效果最好的模型
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

        # 打印当前 epoch 的损失
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            # print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'LR: {current_lr:.4e}')

    # 模型保存
    current_time = datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")
    best_model_path = f"{os.path.join(save_dir, bestmodel_path)}/{bestmodel_path}-{time_str}.pth"

    if best_model is not None:
        torch.save(best_model.state_dict(), best_model_path)
        print(f"Best model is saved at '{best_model_path}'.")
    else:
        print("No improvement in validation loss, model is not saved.")

    return best_model, train_losses, val_losses, time_str


def loss_plot(width_cm, height_cm, train_losses, val_losses, time_str, series_path, i=0):

    save_dir = './model_log/loss_plot'
    _ensure_dir(os.path.join(save_dir, series_path))

    # 可视化损失曲线
    cm_to_inch = 1/2.54  # 厘米转英寸的转换因子
    width_inch = width_cm * cm_to_inch
    height_inch = height_cm * cm_to_inch

    plt.figure(figsize=(width_inch, height_inch), dpi=600)
    plt.plot(train_losses[i:], label='Train Loss', linewidth=0.8)
    plt.plot(val_losses[i:], label='Validate Loss', linewidth=0.8)

    plt.title(series_path, fontsize=9, pad=5)
    plt.xlabel('Epoch', fontsize=9, labelpad=3)
    plt.ylabel('Loss', fontsize=9, labelpad=3)

    # 使用科学计数法格式化y轴
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    # 获取当前坐标轴并设置科学计数法格式
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(8)  # 设置科学计数法偏移值的字体大小
    ax.yaxis.get_offset_text().set_position((0, 1.02))
    # 设置刻度标签格式为科学计数法
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)

    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.legend(fontsize=8)
    legend = ax.legend(fontsize=8, loc='upper right', framealpha=0.6)
    legend.get_frame().set_linewidth(0.5)
    plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    # 计算图表高度范围
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    #  添加最小值标记
    min_val_loss = min(val_losses)
    min_index = val_losses.index(min_val_loss)

    # 选项2：点上方标注
    text_x = min_index
    text_y = min_val_loss + y_range * 0.15  # 点上方15%图表高度

    ax.plot(min_index, min_val_loss, '*', markersize=3, color='red')

    ax.annotate(f'Min Loss: {min_val_loss:.4f}',
                xy=(min_index, min_val_loss),
                xytext=(text_x, text_y),
                fontsize=6, color='red', ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.5,
                                connectionstyle="arc3,rad=0.2"))

    # 调整布局
    plt.tight_layout()

    # 保存为SVG格式
    plt.savefig(
        f"{os.path.join(save_dir, series_path)}/Loss-{time_str}.svg",
        format='svg',
        dpi=600,
        bbox_inches='tight'
    )
    plt.show()


def get_predictions(model, data_loader, device='cuda'):
    """提取训练集、验证集和测试集的真实值和预测值"""
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            # x = x.unsqueeze(1)
            output = model(x)

            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(output.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    return y_true.flatten(), y_pred.flatten()


def evaluate_model(y_true, y_pred):
    """计算回归任务的评估指标

        r_value -- Pearson相关系数;
        rmse -- 均方根误差;
        r2 -- 决定系数;
        rpiq -- RPIQ指标;
        mae -- 平均绝对误差;
    """

    # 计算 Pearson 相关系数
    r_value, _ = pearsonr(y_true, y_pred)

    # 计算 RMSE
    rmse = root_mean_squared_error(y_true, y_pred)
    # rmse = np.sqrt(np.mean((y_true.flatten() - y_pred.flatten())**2))

    # 计算 R²
    r2 = r2_score(y_true, y_pred)

    # 计算 RPIQ
    iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
    rpiq = iqr / rmse

    # 计算 MAE
    mae = np.mean(np.abs(y_true - y_pred))

    return r_value, rmse, r2, rpiq, mae


def correlation_scatter(width_cm, height_cm, y_test_true, y_test_pred, figure_path, time_str):
    save_dir = '../results/model_log/correlation_scatter'
    # label_figure_dir = f'Fig_{series_path[0]}'  # current_path = (label_name,series_path)
    _ensure_dir(save_dir)

    # 创建图形布局
    cm_to_inch = 1/2.54  # 厘米转英寸的转换因子
    width_inch = width_cm * cm_to_inch
    height_inch = height_cm * cm_to_inch
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    fig.set_dpi(600)  # 设置DPI为600

    # 计算测试集指标
    r_value_test, rmse_test, r2_test, rpiq_test, mae_test = evaluate_model(y_test_true, y_test_pred)

    # 计算绝对误差
    errors_test = np.abs(y_test_true - y_test_pred)

    # 创建散点图
    scatter = ax.scatter(
        y_test_true,
        y_test_pred,
        c=errors_test,
        cmap='plasma',  # 使用更清晰的配色方案
        alpha=0.6,
        s=10,            # 适当增大点的大小
        edgecolors="#0b164b",   # 添加白色边框使点更清晰
        linewidths=0.2
    )

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Absolute Error', fontsize=9)  # 颜色条标签字体大小设为9
    cbar.ax.tick_params(labelsize=8)  # 颜色条刻度标签字体大小设为8

    # 添加理想线
    min_val = min(np.min(y_test_true), np.min(y_test_pred))
    max_val = max(np.max(y_test_true), np.max(y_test_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, alpha=0.7)

    # 添加回归线
    sns.regplot(
        x=y_test_true.flatten(),
        y=y_test_pred.flatten(),
        scatter=False,
        ax=ax,
        color='red',
        line_kws={"linewidth": 1, "linestyle": "-", "alpha": 0.8}
    )

    # 设置标题和标签
    ax.set_title('True vs Predicted Values', fontsize=9, pad=5)
    ax.set_xlabel('True', fontsize=9, labelpad=5)
    ax.set_ylabel('Predicted', fontsize=9, labelpad=5)

    # 设置坐标轴刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=8)

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)

    # 显示统计指标（使用更清晰的格式）
    metrics_text = (
        f'Pearson r = {r_value_test:.4f}\n'
        f'R² = {r2_test:.4f}\n'
        f'RMSE = {rmse_test:.4f}\n'
        # f'MAE = {mae_test:.4f}\n'
        f'RPIQ = {rpiq_test:.4f}'
    )

    # 将指标框放在左上角
    ax.text(
        0.05, 0.95,
        metrics_text,
        transform=ax.transAxes,
        fontsize=6,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.6,
            edgecolor='gray',
            pad=0.5,
            linewidth=0.5,
            linestyle='--',       # 边框样式（实线/虚线等）
            capstyle='round'      # 边框端点样式
        )
    )

    # 添加数据点数量信息
    n_samples = len(y_test_true.flatten())
    ax.text(
        0.95, 0.05,
        f'n = {n_samples}',
        transform=ax.transAxes,
        fontsize=6,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.6,
            edgecolor='gray',
            pad=0.2,
            linewidth=0.5,
            linestyle='--',       # 边框样式（实线/虚线等）
            capstyle='round'      # 边框端点样式
        )
    )

    # 调整布局
    plt.tight_layout()

    # 保存图像
    fig.savefig(
        f"{save_dir}/{figure_path}.svg",
        format='svg',
        dpi=600,
        bbox_inches='tight'
    )

    plt.show()


def TPcomparison(y_train_true, y_train_pred, y_val_true, y_val_pred, y_test_true, y_test_pred):

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=300)

    # 绘制训练集的真实值与预测值对比
    axes[0].plot(y_train_true, label='Training True', color='blue', marker='o', alpha=0.7)
    axes[0].plot(y_train_pred, label='Training Predicted', color='red', marker='*', alpha=0.7)
    axes[0].set_title('Training Set')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Value')
    axes[0].legend()

    # 绘制验证集的真实值与预测值对比
    axes[1].plot(y_val_true, label='Validation True', color='blue', marker='o', alpha=0.7)
    axes[1].plot(y_val_pred, label='Validation Predicted', color='red', marker='*', alpha=0.7)
    axes[1].set_title('Validation Set')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Value')
    axes[1].legend()

    # 绘制测试集的真实值与预测值对比
    axes[2].plot(y_test_true, label='Test True', color='blue', marker='o', alpha=0.7)
    axes[2].plot(y_test_pred, label='Test Predicted', color='red', marker='*', alpha=0.7)
    axes[2].set_title('Test Set')
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Value')
    axes[2].legend()

    # 调整布局，使子图之间不重叠
    plt.tight_layout()

    # 显示图像
    plt.show()
