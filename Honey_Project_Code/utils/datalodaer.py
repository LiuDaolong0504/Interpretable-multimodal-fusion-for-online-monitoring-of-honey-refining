import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def data_split(spectra_path, label_path, test_size=0.4, random_state=42):
    """1.用于光谱数据读取, 数据集划分, 不涉及融合

    Args:
        spectra_path (str): _description_
        label_path (str): _description_
        test_size (float, optional): _description_. Defaults to 0.4.
        random_state (int, optional): _description_. Defaults to 42.

    Returns:
        tuple: (train_set, val_set, test_set), 每个set包含(X, y, idx)
    """
    spectra_df = pd.read_csv(spectra_path)
    label_df = pd.read_csv(label_path)
    # 使用merge函数进行内连接（取交集）
    merged_df = pd.merge(label_df.iloc[:, 1:3], spectra_df.iloc[:, 1:], on="名称", how="inner")

    features = merged_df.iloc[:, 2:2077].dropna(axis=1).to_numpy(dtype=np.float32)
    label2train = merged_df.iloc[:, 1:2].values.astype(np.float32).flatten()
    indices = merged_df["名称"].tolist()

    # print(f"光谱数据形状: {spectra.shape}")
    # print(f"标签数据形状: {label2train.shape}")

    # 先划分 train(0.6) 和 temp(0.4)
    X_train, X_temp, y_train, y_temp, indices_train, indices_temp = train_test_split(
        features, label2train, indices, test_size=test_size, random_state=random_state)

    # 再把 temp 划成 val(0.2) 和 test(0.2)
    # temp 占原始 0.4，这里 test_size=0.5 -> 0.2/0.2
    X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(X_temp, y_temp, indices_temp, test_size=0.5, random_state=random_state)

    train_set = (X_train, y_train, indices_train)
    val_set = (X_val, y_val, indices_val)
    test_set = (X_test, y_test, indices_test)
    return train_set, val_set, test_set


def fused_data_split(spectra_path, image_features_path, label_path, test_size=0.4, random_state=42):
    """2.用于光谱+图像数据读取, 数据集划分; 不涉及融合(融合方式在训练代码中显式定义)

    Args:
        spectra_path (str): _description_
        image_features_path (str): _description_
        label_path (str): _description_
        test_size (float, optional): _description_. Defaults to 0.4.
        random_state (int, optional): _description_. Defaults to 42.

    Returns:
        tuple: (train_set, val_set, test_set), 每个set包含(X_spec, X_image, y, idx)
    """

    # --------------- 1. 读取光谱、RGB特征、标签三个文件 ---------------
    spectra_df = pd.read_csv(spectra_path)
    image_df = pd.read_csv(image_features_path)
    label_df = pd.read_csv(label_path)

    # --------------- 2. 合并数据 使用merge函数进行内连接(Inner Join 确保样本对应,取交集) ---------------
    # 合并顺序： Label->spectra->image_feature
    merged_df_temp = pd.merge(label_df.iloc[:, 1:3], spectra_df.iloc[:, 1:], on="名称", how="inner")
    merged_df = pd.merge(merged_df_temp, image_df, on="名称", how="inner")

    # --------------- 3. 提取数据矩 ---------------
    # 提取标签 y
    y = merged_df.iloc[:, 1:2].values.astype(np.float32).flatten()
    # 提取 光谱 特征 (光谱数值列)
    X_spec = merged_df.iloc[:, 2:2077].values.astype(np.float32)
    # 提取 图像 特征 (剩余的数值列)
    X_image = merged_df.iloc[:, 2077:].values.astype(np.float32)
    # 提取 索引  ('名称'的数值列)
    indices = merged_df['名称']

    # --------------- 4. 数据划分 ---------------
    # 第一次划分: Train vs Temp
    X_spec_train, X_spec_temp, X_image_train, X_image_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X_spec, X_image, y, indices, test_size=test_size, random_state=random_state
    )

    # 第二次划分: Val vs Test
    X_spec_val, X_spec_test, X_image_val, X_image_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_spec_temp, X_image_temp, y_temp, idx_temp, test_size=0.5, random_state=random_state
    )
    return (X_spec_train, X_image_train, y_train, idx_train), \
           (X_spec_val, X_image_val, y_val, idx_val), \
           (X_spec_test, X_image_test, y_test, idx_test)
