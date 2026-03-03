import optuna
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from joblib import parallel_backend  # 用于并行加速，避免警告


def tune_pls(X_train, y_train, preprocessor=None):
    """使用网格搜索和5折交叉验证训练PLS

    Args:
        X_train (arr: np.ndarray): 包含训练和验证数据的训练集X
        y_train (arr: np.ndarray): 包含训练和验证数据的训练集y
        preprocessor (Class, optional): 数据预处理方法. Defaults to None.

    Returns:
        best_model->最佳PLS模型, evaluation->最优参数和RMSE结果字典
    """
    # ===== 1. 定义 PLS pipeline：标准化 + PLS =====
    if preprocessor is None:
        pipe = Pipeline([
            ("scaler", StandardScaler()),   # 数据标准化
            ("pls", PLSRegression())        # PLS 回归模型
        ])
    else:
        steps = [
            ("preprocess", preprocessor),  # 动态插入预处理
            ("scaler", StandardScaler()),  # 为了保持与之前一致，保留标准化
            ("pls", PLSRegression())      # PLS 模型
        ]
        pipe = Pipeline(steps)

    # ===== 2. 使用 5 折交叉验证调参 n_components =====
    param_grid = {
        "pls__n_components": list(range(1, 50))  # 尝试不同的主成分数
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # ===== 3. 使用 GridSearchCV 进行超参数搜索 =====
    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",  # 最小化 RMSE（由于默认是负值，所以得分越大越好）
        n_jobs=-1,  # 使用所有 CPU 核心
        verbose=0   # 输出训练进度
    )

    # 为了避免 joblib 警告，显式地使用 parallel_backend
    with parallel_backend("threading", n_jobs=-1):  # 使用多线程加速
        search.fit(X_train, y_train.ravel())  # 训练模型

    # 输出最优参数和结果
    # print("CV 最优参数:", search.best_params_)
    # print("CV 最优 RMSE:", -search.best_score_)
    evaluation = {
        "CV 最优参数:": int(search.best_params_['pls__n_components']),
        "CV 最优 RMSE:": round(float(-search.best_score_), 4)
    }
    # 获取最佳模型
    # best_model = search.best_estimator_
    return search.best_estimator_, evaluation


def tune_xgb_optuna(train_set, val_set, n_trials=100, timeout=None, random_state=42):
    """
    使用 Optuna 对 XGBRegressor 进行贝叶斯超参数优化
    ------------------------------------------------
    参数：
        train_set   : 元组 (X_train, y_train)
        val_set     : 元组 (X_val, y_val)
        n_trials    : 最大搜索次数（trial 数量）
        timeout     : 允许的最大搜索时间（秒），为 None 时不限制时间
        random_state: 随机种子，保证实验可复现

    返回：
        best_model  : 验证集 RMSE 最优的 XGBRegressor 模型（已经在 train+val 上重新训练）
        study       : Optuna 的 Study 对象（包含所有 trial 记录，可用于可视化分析）
        evaluation  : 字典，包含最优参数、最优 RMSE、最优 trial 等信息
    """

    # 解包训练集和验证集
    X_train, y_train= train_set
    X_val, y_val= val_set

    # 将 y 转为一维数组，避免某些情况下的形状问题
    y_train = np.array(y_train).ravel()
    y_val = np.array(y_val).ravel()

    # =========================
    #  1. 定义目标函数（Objective）
    # =========================
    def objective(trial):
        """
        Optuna 目标函数：
        给定一组 trial 超参数，训练一个 XGBRegressor，
        然后在验证集上计算 RMSE 作为优化目标（越小越好）。
        """

        # ---------- 1.1 定义超参数搜索空间 ----------
        # 学习率：对数均匀分布，低学习率通常更稳
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)

        # 树的最大深度：越深越复杂，容易过拟合
        max_depth = trial.suggest_int("max_depth", 2, 8)

        # 最小子节点权重：控制叶子节点最小样本量，越大越保守
        min_child_weight = trial.suggest_float("min_child_weight", 1.0, 10.0)

        # 行采样比例：控制每棵树使用多少样本，<1 有助于防止过拟合
        subsample = trial.suggest_float("subsample", 0.5, 1.0)

        # 列采样比例：控制每棵树使用多少特征
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.3, 1.0)

        # L2 正则（lambda）：>0 有助于防止过拟合
        reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True)

        # L1 正则（alpha）：>0 可以促使部分特征权重趋近于 0
        reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)

        # gamma：拆分节点所需的最小损失下降值，越大越保守
        gamma = trial.suggest_float("gamma", 0.0, 5.0)

        # ---------- 1.2 构建模型 ----------
        model = XGBRegressor(
            n_estimators=5000,             # 设置较大上限，配合 early stopping 使用
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            gamma=gamma,
            objective="reg:squarederror",
            random_state=random_state,
            tree_method="hist",            # 一般用 hist 更快更稳
            eval_metric="rmse",
            early_stopping_rounds=100,     # 100 轮无提升则提前停止
        )

        # ---------- 1.3 在训练集上拟合，并使用验证集做 early stopping ----------
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # ---------- 1.4 在验证集上计算 RMSE ----------
        y_val_pred = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_val_pred)

        # 将当前最优迭代轮数记录到 trial 的 user_attrs 中（可选，用于后续分析）
        trial.set_user_attr("best_iteration", int(model.best_iteration))

        return rmse

    # =========================
    #  2. 创建 Study，并运行优化
    # =========================

    # 建议显式设置采样器为 TPE，方向为“最小化”

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        study_name="xgb_optuna_tuning",
        direction="minimize",   # RMSE 越小越好
        sampler=sampler
    )

    # 开始搜索：可以设置 n_trials 和/或 timeout
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True  # 显示进度条
    )

    # =========================
    #  3. 取出最优结果，并在 train+val 上重新训练最终模型
    # =========================

    best_trial = study.best_trial
    best_params = best_trial.params
    best_rmse = best_trial.value
    best_iter = best_trial.user_attrs.get("best_iteration", None)

    print("Optuna 调参完成：")
    print("  最优 trial 编号:", best_trial.number)
    print("  最优参数:", best_params)
    print("  最优验证集 RMSE:", best_rmse)
    if best_iter is not None:
        print("  对应 best_iteration:", best_iter)

    # 将训练集与验证集合并，用于重新训练最终模型（也可以只用训练集，这里按需求自行选择）
    # X_train_full = np.concatenate([X_train, X_val], axis=0)
    # y_train_full = np.concatenate([y_train, y_val], axis=0)

    # 使用最优参数构建最终模型
    best_model = XGBRegressor(
        n_estimators=5000,
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        min_child_weight=best_params["min_child_weight"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        reg_lambda=best_params["reg_lambda"],
        reg_alpha=best_params["reg_alpha"],
        gamma=best_params["gamma"],
        objective="reg:squarederror",
        random_state=random_state,
        tree_method="hist",
        eval_metric="rmse",
        early_stopping_rounds=100,
    )

    # 在 train+val 上重新训练最终模型
    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # =========================
    #  4. 整理返回信息
    # =========================

    # 将所有 trial 转为 DataFrame，方便保存与分析
    # history_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))

    evaluation = {
        "最优参数": best_params,
        "最优验证RMSE": best_rmse,
        "最优trial编号": best_trial.number,
        "最优迭代轮数": best_iter,
        # "搜索记录": history_df,
    }

    return best_model, evaluation
