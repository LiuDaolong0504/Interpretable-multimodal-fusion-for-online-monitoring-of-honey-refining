# from typing import Optional, Tuple, Union
import os
# import torch
# from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# from datetime import datetime
import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
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


def correlation_scatter(width_cm, height_cm, y_test_true, y_test_pred, figure_name):
    """绘制相关性散点图

    Args:
        width_cm (float): 图形宽度，单位：cm
        height_cm (float): 图形高度，单位：cm
        y_test_true (nparray): 测试集真实label
        y_test_pred (nparray): 测试集预测label
        figure_name (str): 图像保存名称
    """
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
        f"{save_dir}/{figure_name}.svg",
        format='svg',
        dpi=600,
        bbox_inches='tight'
    )

    plt.show()


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
