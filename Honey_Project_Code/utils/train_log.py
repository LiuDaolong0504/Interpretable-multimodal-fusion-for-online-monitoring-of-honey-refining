from datetime import datetime
import numpy as np
import os
import pandas as pd


def _ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


class EvaluationMetricsLogger:
    """
    评估指标日志记录器，支持累积保存多个模型的评估结果
    """

    def __init__(self):
        log_file = "evaluation_metrics_log.csv"
        self.existing_data = None

        save_dir = '../results/model_evaluation'
        _ensure_dir(os.path.join(save_dir,))
        self.log_file = os.path.join(save_dir, log_file)

        # 如果文件已存在，加载现有数据
        if os.path.exists(self.log_file):
            self.existing_data = pd.read_csv(self.log_file)
            print(f"加载现有日志文件: {self.log_file}，包含 {len(self.existing_data)} 条记录")

    def log_metrics(self, metrics_data: dict, model_name: str = " ", label_name: str = " ", parameters=None):
        """记录一次实验的评估指标

        Args:
            metrics_data (dict): 保存评估指标的字典
            model_name (str, optional): 模型名称. Defaults to " ".
            label_name (str, optional): 标签名称. Defaults to " ".
            parameters (dict, optional): _description_. Defaults to None.
        """
        # 添加参数信息（如果提供）
        if parameters is not None:
            for key, value in parameters.items():
                metrics_data[f'{key}'] = [value, 0, 0]

        # 创建当前实验的DataFrame
        current_df = pd.DataFrame(metrics_data)

        # 合并到现有数据
        if self.existing_data is not None:
            combined_df = pd.concat([self.existing_data, current_df], ignore_index=True)
        else:
            combined_df = current_df

        # 保存到文件
        combined_df.to_csv(self.log_file, index=False, encoding='utf-8-sig')
        self.existing_data = combined_df

        print(f"实验记录已保存: {label_name}-{model_name}")
        print(f"训练集 R²: {metrics_data['R2'][0]:.4f}, 验证集 R²: {metrics_data['R2'][1]:.4f}, 测试集 R²: {metrics_data['R2'][2]:.4f}\n")


class PredictionSaver:
    """预测结果保存器"""

    def __init__(self):
        predict_file = "predictions_log.csv"
        self.existing_data = None

        save_dir = '../results/model_evaluation'
        _ensure_dir(os.path.join(save_dir,))
        self.log_file = os.path.join(save_dir, predict_file)

        # 如果文件已存在，加载现有数据
        if os.path.exists(self.log_file):
            self.existing_data = pd.read_csv(self.log_file)
            print(f"加载现有预测文件: {self.log_file}，包含 {len(self.existing_data)} 条预测记录")

    def save_prediction(self, results_df):
        """保存预测结果到csv

        Args:
            results_df (DataFrame): 结果DataFrame
        """
        # 创建当前实验的DataFrame
        current_df = results_df

        # 合并到现有数据
        if self.existing_data is not None:
            combined_df = pd.concat([self.existing_data, current_df], ignore_index=True)
        else:
            combined_df = current_df

        # 保存到文件
        combined_df.to_csv(self.log_file, index=False, encoding='utf-8-sig')
        self.existing_data = combined_df

        print(f"预测结果已保存OK")
