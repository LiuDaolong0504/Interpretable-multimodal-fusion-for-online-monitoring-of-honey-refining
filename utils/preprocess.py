import numpy as np
from scipy.signal import savgol_filter, detrend
from sklearn.base import BaseEstimator, TransformerMixin


class SNV(BaseEstimator, TransformerMixin):
    """标准正态变量变换 (Standard Normal Variate)"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # 沿光谱轴(axis=1)计算均值和标准差
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        return (X - mean) / std


class MSC(BaseEstimator, TransformerMixin):
    """多元散射校正 (Multiplicative Scatter Correction)"""

    def __init__(self):
        self.ref_spectrum = None

    def fit(self, X, y=None):
        # 使用训练集的平均光谱作为参考光谱
        self.ref_spectrum = np.mean(X, axis=0)
        return self

    def transform(self, X, y=None):
        X_msc = np.zeros_like(X)
        for i in range(X.shape[0]):
            # 对每个样本进行线性回归: x_i = k * x_ref + b
            # polyfit(x, y, 1) 返回 [k, b]
            fit = np.polyfit(self.ref_spectrum, X[i, :], 1)
            k = fit[0]
            b = fit[1]
            # 校正: (x_i - b) / k
            X_msc[i, :] = (X[i, :] - b) / k
        return X_msc


class SavitzkyGolay(BaseEstimator, TransformerMixin):
    """Savitzky-Golay 平滑/求导"""

    def __init__(self, window_length=15, polyorder=3, deriv=1):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # 对每一行（每个样本）应用 SG 滤波
        return savgol_filter(X, self.window_length, self.polyorder, deriv=self.deriv, axis=1)


class Detrend(BaseEstimator, TransformerMixin):
    """
    基线校正：去趋势 (Detrending)
    通常用于消除由于粉末装填不实或光散射引起的光谱基线倾斜。
    默认去除线性趋势 (linear)。
    """

    def __init__(self, type='linear'):
        self.type = type  # 'linear' or 'constant'

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # scipy.signal.detrend 默认对 axis=-1 (即最后一个轴) 进行操作，这正是我们需要的
        return detrend(X, axis=1, type=self.type)


class DummyTransformer(BaseEstimator, TransformerMixin):
    """不做任何处理，作为对照组"""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X
