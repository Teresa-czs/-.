# coding=utf-8
"""
两阶段智能推荐 Python 代码 （代价收益 + 拉格朗日松弛）
- 第一阶段：代价收益算法求 Z_i（项目推荐次数分布）
- 第二阶段：拉格朗日松弛 + 次梯度 + 贪心修复 求解推荐矩阵 x_{u,i}
  * 将每个项目的容量约束 ∑_u x_{ui} ≤ Z_i 做拉格朗日松弛，引入 λ_i ≥ 0
  * 对给定 λ，每个用户的子问题是：从所有项目中选 N 个最大化 R_ui - λ_i
  * 用次梯度法 **最小化对偶函数 L(λ)**，L(λ) 为原最大化问题的上界
  * 最后在最优 λ 上用一个简单的贪心算法构造原问题的可行解
- 最终在得到最佳推荐矩阵 x_{ui} 后：
  * 计算原问题目标值 ∑_{u,i} R_ui x_ui，作为下界
  * 使用拉格朗日对偶最优值作为上界，输出 (上界 - 下界) / 下界 作为 gap
  * 同时在迭代过程中记录“历次迭代的最佳上界”，并画出其随迭代次数变化的曲线（dual_history.png）

注意：
- 已忽略论文中的决策代价约束（m_i, r_i 相关部分）
- 评分过滤阈值默认为 0
- 本实现不依赖外部 MIP 求解器（Gurobi / CBC），运行速度比列生成 + MIP 快很多
- 会画出 dual 迭代曲线（历次最优上界），保存在 dual_history.png

依赖：
pip install numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import math
import time

# 画图用
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# 兼容保留（当前实现不依赖）
try:
    import pulp  # noqa: F401
except ImportError:
    pulp = None

try:
    import gurobipy as gp  # noqa: F401
    from gurobipy import GRB  # noqa: F401
except ImportError:
    gp = None

# ---------------------------
# 全局参数
# ---------------------------
PERCENT = 0.9    # 第一阶段：分布多样性 vs 重要性 权重 β
N = 10           # 每个用户推荐 N 个项目

# 根据你的路径配置
PATH_FEATURE_IMPORTANCE = r"E:\科研\深度学习框架--推荐系统\金融科技推荐\feature_importance_reordered.csv"
PATH_USER_ASSET_PREF   = r"E:\科研\深度学习框架--推荐系统\金融科技推荐\user_asset_preferences_full.csv"


# =========================================================
# 数据读取
# =========================================================
def load_data():
    """
    读取用户-项目评分矩阵 A 和项目重要性向量 B
    A: shape = (num_items, num_users)
    B: shape = (num_items,)
    """
    A_raw = pd.read_csv(PATH_USER_ASSET_PREF, header=None).to_numpy()
    # 你的原始数据是转置+去掉首行首列
    A_raw = A_raw.T
    A = A_raw[1:, 1:].astype(float)

    B_raw = pd.read_csv(PATH_FEATURE_IMPORTANCE, header=None).to_numpy()
    B = B_raw[1:, 1].astype(float)

    return A, B


# =========================================================
# 第一阶段：代价收益算法（来自第一篇论文）
# =========================================================
def build_earning_curve(N, num_users):
    """
    构造“收益曲线” earning_iter[t]，对应 z 从 t-1 -> t 推荐次数的增量收益。
    对应论文中 Δ_z_i 的离散化近似。
    """
    iteration_coef = np.zeros(num_users + 1)
    earning = np.zeros(num_users + 1)

    for t in range(1, num_users + 1):
        z = t / (N * num_users)
        iteration_coef[t] = -z * math.log(z)

    earning[1] = iteration_coef[1]
    for t in range(2, num_users + 1):
        earning[t] = iteration_coef[t] - iteration_coef[t - 1]

    return iteration_coef, earning


def subproblem_Z_2_1_new(N, lamda, earning_iter, A, percent=PERCENT):
    """
    第一阶段子问题：
    给定 λ_i（由项目重要性缩放得到），
    使用“代价收益迭代”在总推荐次数 N*U 不变下调整 Z_i 分布，
    目标：percent * 分布多样性 + (1 - percent) * 项目重要性。
    """
    num_items, num_users = A.shape
    Z = np.zeros(num_items, dtype=int)

    # 初始：前 N 个项目分配满所有用户
    Z[:N] = num_users
    LARGE = 1e8

    def objective(Z_vec):
        total = 0.0
        for t in range(num_items):
            if Z_vec[t] != 0:
                z_norm = Z_vec[t] / (N * num_users)
                total += -percent * z_norm * math.log(z_norm) + (1 - percent) * Z_vec[t] * lamda[t]
        return total

    min_w = objective(Z)

    Zi_price_coef = np.full(num_items, LARGE, dtype=float)
    Zi_benefit_coef = np.full(num_items, -LARGE, dtype=float)

    for t in range(num_items):
        Zi_number = Z[t]
        if Zi_number != 0:
            Zi_price_coef[t] = percent * earning_iter[Zi_number] + (1 - percent) * lamda[t]
        if Zi_number != num_users:
            Zi_benefit_coef[t] = percent * earning_iter[Zi_number + 1] + (1 - percent) * lamda[t]

    minValue = Zi_price_coef.min()
    minIndex = Zi_price_coef.argmin()
    maxValue = Zi_benefit_coef.max()
    maxIndex = Zi_benefit_coef.argmax()

    number_die = 0
    while maxValue > minValue:
        number_die += 1
        min_w += maxValue - minValue

        # 从 minIndex 项目减少一次推荐
        Z[minIndex] -= 1
        bb = Z[minIndex]
        if bb != num_users:
            Zi_benefit_coef[minIndex] = percent * earning_iter[bb + 1] + (1 - percent) * lamda[minIndex]
        else:
            Zi_benefit_coef[minIndex] = -LARGE

        # 给 maxIndex 项目增加一次推荐
        Z[maxIndex] += 1
        cc = Z[maxIndex]
        if cc != 0:
            Zi_price_coef[maxIndex] = percent * earning_iter[cc] + (1 - percent) * lamda[maxIndex]
        else:
            Zi_price_coef[maxIndex] = LARGE

        Zi_price_coef[minIndex] = (
            percent * earning_iter[bb] + (1 - percent) * lamda[minIndex]
            if Z[minIndex] != 0 else LARGE
        )
        Zi_benefit_coef[maxIndex] = (
            percent * earning_iter[cc + 1] + (1 - percent) * lamda[maxIndex]
            if Z[maxIndex] != num_users else -LARGE
        )

        minValue = Zi_price_coef.min()
        minIndex = Zi_price_coef.argmin()
        maxValue = Zi_benefit_coef.max()
        maxIndex = Zi_benefit_coef.argmax()

    min_w = objective(Z)
    return Z, min_w, number_die


def stage1_cost_benefit_distribution(A, B, N):
    """
    第一阶段：
    - 先用 earning curve 计算 Δ_z
    - 再把项目重要性 B 映射到和 earning 同量级，得到 λ_i
    - 调用子问题得到 Z_i
    """
    num_items, num_users = A.shape
    _, earning_iter = build_earning_curve(N, num_users)

    e_max = earning_iter[1:].max()
    e_min = earning_iter[1:].min()
    lamda_max = B.max()
    lamda_min = B.min()

    if lamda_max - lamda_min == 0:
        lamda_scaled = np.full_like(B, (e_max + e_min) / 2)
    else:
        lamda_scaled = (B - lamda_min) * (e_max - e_min) / (lamda_max - lamda_min) + e_min

    Z, min_w, number_die = subproblem_Z_2_1_new(N, lamda_scaled, earning_iter, A, PERCENT)
    print(f"第一阶段完成：目标值 = {min_w:.4f}，迭代交换次数 = {number_die}")
    return Z, min_w, number_die, lamda_scaled


# =========================================================
# 第二阶段：拉格朗日松弛 + 次梯度
# =========================================================
def greedy_construct_primal(A, Z_value, N, lambda_vec, rating_threshold=0.0):
    """
    给定 λ 向量，用贪心算法构造原问题的一个可行解：
        max ∑_{u,i} R_ui x_ui
        s.t. ∑_i x_ui = N,          ∀u
             ∑_u x_ui ≤ Z_i,        ∀i
             x_ui ∈ {0,1}
    算法：
        - 按 “有效评分” score = R_ui - λ_i 从大到小枚举 (u,i) 对
        - 若该对不违反用户/项目容量约束，则选用
        - 最后若仍有用户推荐数 < N，则在剩余容量下用原始评分 R_ui 再补齐
    """
    num_items, num_users = A.shape
    x = np.zeros_like(A, dtype=int)
    user_cnt = np.zeros(num_users, dtype=int)
    item_cnt = np.zeros(num_items, dtype=int)

    # Step 1: 按 R_ui - λ_i 排序所有 (u,i) 对
    pairs = []
    for i in range(num_items):
        for u in range(num_users):
            rating = A[i, u]
            if rating < rating_threshold:
                continue
            score = rating - float(lambda_vec[i])
            pairs.append((score, u, i))

    if not pairs:
        # 极端情况：所有评分都被过滤掉，返回全 0
        return x

    pairs.sort(key=lambda t: t[0], reverse=True)

    # Step 2: 贪心选取不违反容量的 (u,i)
    for score, u, i in pairs:
        if user_cnt[u] >= N:
            continue
        if item_cnt[i] >= Z_value[i]:
            continue
        x[i, u] = 1
        user_cnt[u] += 1
        item_cnt[i] += 1
        if np.all(user_cnt >= N):
            break

    # Step 3: 如果还有用户没被推荐满 N 个，用原始评分在剩余容量下补齐
    if np.any(user_cnt < N):
        for u in range(num_users):
            while user_cnt[u] < N:
                best_i = None
                best_rating = -1e18
                for i in range(num_items):
                    if item_cnt[i] >= Z_value[i]:
                        continue
                    if x[i, u] == 1:
                        continue
                    rating = A[i, u]
                    if rating < rating_threshold:
                        continue
                    if rating > best_rating:
                        best_rating = rating
                        best_i = i
                if best_i is None:
                    # 没有剩余可用项目了，只能提前结束
                    break
                x[best_i, u] = 1
                user_cnt[u] += 1
                item_cnt[best_i] += 1

    return x


def lagrangian_relaxation_stage2(A,
                                 Z_value,
                                 N,
                                 rating_threshold=0.0,
                                 max_iter=200,
                                 step_size=5.0,
                                 step_decay=0.95,
                                 tol_rel_dual=1e-4,
                                 verbose=True):
    """
    第二阶段：对项目容量约束做拉格朗日松弛，用次梯度法 **最小化对偶函数 L(λ)**。

    原模型（简化后）：
        max ∑_{u,i} R_ui x_ui
        s.t. ∑_i x_ui = N,          ∀u
             ∑_u x_ui ≤ Z_i,        ∀i
             x_ui ∈ {0,1}

    对约束 ∑_u x_ui ≤ Z_i 引入拉格朗日乘子 λ_i ≥ 0，得到对偶函数：
        L(λ) = max_x ∑_{u,i} R_ui x_ui + ∑_i λ_i (Z_i - ∑_u x_ui)
             = ∑_i λ_i Z_i + ∑_u max_{|S_u|=N} ∑_{i∈S_u} (R_ui - λ_i)

    - 给定 λ，每个用户 u 的子问题是：从所有项目中挑 N 个，使 R_ui - λ_i 最大。
    - 由于原问题是“最大化”，L(λ) 是上界，**我们要最小化 L(λ)**。
    - 次梯度 g_i = Z_i - ∑_u x_ui（L 对 λ_i 的次梯度）
      更新：λ^{k+1} = [λ^k - α_k * g]^+  （投影到 λ ≥ 0）
    - 收敛判停：
      * ||g|| 很小（接近可行）
      * 或两个连续迭代的 dual 相对变化 ≤ tol_rel_dual（默认 0.01%）

    额外功能：
    - 在迭代过程中记录“历次迭代的最佳上界”（best_dual_history），
      并在结束后画出 best_dual_history 随迭代次数的变化曲线，保存为 dual_history.png。
    """
    num_items, num_users = A.shape
    lambda_vec = np.zeros(num_items, dtype=float)
    eligible_mask = (A >= rating_threshold)

    best_dual = None          # 记录“最小”的对偶上界
    best_lambda = lambda_vec.copy()

    # 每个用户最多只能推荐 min(N, num_items) 个项目
    K = min(N, num_items)

    dual_history = []         # 当前迭代的对偶值（可选）
    best_dual_history = []    # 历次迭代的最佳上界

    dual_prev = None

    for k in range(max_iter):
        # -------- 解拉氏子问题：给定 λ，逐个用户选 N 个项目 --------
        x_relaxed = np.zeros_like(A, dtype=int)
        # L(λ) 的那一部分：∑_i λ_i Z_i
        dual_obj = float(np.dot(lambda_vec, Z_value))

        for u in range(num_users):
            scores = A[:, u] - lambda_vec
            # 应用评分过滤阈值
            scores_masked = np.where(eligible_mask[:, u], scores, -1e12)

            if K <= 0:
                continue
            if K >= num_items:
                top_idx = np.arange(num_items)
            else:
                # argpartition 复杂度 O(I)，适合大规模
                top_idx = np.argpartition(-scores_masked, K - 1)[:K]
                # 再对前 K 个做一次排序（非必须，但更稳定）
                top_idx = top_idx[np.argsort(-scores_masked[top_idx])]

            x_relaxed[top_idx, u] = 1
            dual_obj += float(np.sum(scores[top_idx]))

        dual_history.append(dual_obj)

        # -------- 记录目前最好的（最小）对偶上界 --------
        if best_dual is None or dual_obj < best_dual:
            best_dual = dual_obj
            best_lambda = lambda_vec.copy()
        best_dual_history.append(best_dual)

        # -------- 计算次梯度并打印信息 --------
        # g_i = Z_i - ∑_u x_ui
        g = Z_value - x_relaxed.sum(axis=1)
        norm_g = float(np.linalg.norm(g))

        if verbose:
            print(
                f"[LR] iter {k + 1:3d}: dual = {dual_obj:.4f}, "
                f"best_dual = {best_dual:.4f}, "
                f"||g|| = {norm_g:.4f}, step = {step_size:.4f}"
            )

        # 相对变化判停：|dual_k - dual_{k-1}| / max(1, |dual_{k-1}|) <= 0.01%
        if dual_prev is not None:
            rel_change = abs(dual_obj - dual_prev) / max(1.0, abs(dual_prev))
            if rel_change <= tol_rel_dual:
                if verbose:
                    print(
                        f"[LR] dual 相对变化 {rel_change:.6e} ≤ {tol_rel_dual:.6e}，停止迭代。"
                    )
                break
        dual_prev = dual_obj

        # 次梯度范数很小也可以停
        if norm_g < 1e-6:
            if verbose:
                print("[LR] ||g|| 非常小，认为已接近最优，停止迭代。")
            break

        # -------- 次梯度更新 λ：λ ← [λ - α * g]_+ --------
        if norm_g > 0:
            g_normed = g / norm_g
        else:
            g_normed = g

        lambda_vec = lambda_vec - step_size * g_normed
        lambda_vec = np.maximum(0.0, lambda_vec)  # 投影到 λ ≥ 0

        # 步长逐渐衰减
        step_size *= step_decay

    # -------- 画出“历次最佳上界”的 dual 迭代曲线 --------
    if plt is not None and len(best_dual_history) > 0:
        try:
            plt.figure()
            plt.plot(
                range(1, len(best_dual_history) + 1),
                best_dual_history,
                marker='o'
            )
            plt.xlabel("Iteration")
            plt.ylabel("Best dual value (upper bound)")
            plt.title("Best Lagrangian Dual Upper Bound over Iterations")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("dual_history.png", dpi=200)
            plt.close()
            if verbose:
                print("best dual 上界迭代曲线已保存为 dual_history.png")
        except Exception as e:
            print(f"绘制 dual 迭代图时出错：{e}")
    else:
        if verbose:
            print("未能绘制 dual 迭代图（可能未安装 matplotlib）。")

    # -------- 用“最小对偶上界”对应的 λ 构造原问题可行解 --------
    x_primal = greedy_construct_primal(A, Z_value, N, best_lambda, rating_threshold)

    return x_primal, best_lambda, best_dual


def column_generation_stage2(A,
                             Z_value,
                             N,
                             rating_threshold=0.0,
                             max_iter=200,
                             step_size=5.0,
                             step_decay=0.95):
    """
    为了兼容你原来的 main() 调用接口，这里仍保留函数名
    column_generation_stage2，但内部实现已经完全改为：
        拉格朗日松弛 + 次梯度 + 贪心修复

    额外增加：
    - 使用最终得到的推荐矩阵 x_result 计算原问题目标值 ∑_{u,i} R_ui x_ui，作为下界；
    - 使用拉格朗日对偶最优值 best_dual 作为上界；
    - 输出 gap = (上界 - 下界) / 下界。
    """
    x_result, lambda_best, best_dual = lagrangian_relaxation_stage2(
        A,
        Z_value,
        N,
        rating_threshold=rating_threshold,
        max_iter=max_iter,
        step_size=step_size,
        step_decay=step_decay,
        tol_rel_dual=1e-4,  # 0.01%
        verbose=True
    )

    # 计算原问题的目标值：∑_{u,i} R_ui x_ui，作为下界
    primal_value = float((A * x_result).sum())

    # 计算 gap = (上界 - 下界) / 下界
    if primal_value != 0:
        gap = (best_dual - primal_value) / primal_value
    else:
        gap = float("inf")

    print(f"拉格朗日松弛完成，最小对偶上界（upper bound） ≈ {best_dual:.4f}")
    print(f"原问题可行解的目标值（下界 lower bound） ≈ {primal_value:.4f}")
    if math.isfinite(gap):
        print(
            f"最终最优性 gap = (上界 - 下界) / 下界 ≈ {gap:.6f} "
            f"（约 {gap * 100:.4f}%）"
        )
    else:
        print("下界为 0，无法计算相对 gap。")

    return x_result


# =========================================================
# 结果评估
# =========================================================
def evaluate(A, B_scaled, Z_value, x_result_max, N):
    """
    简单评估：
    - 分布多样性（Shannon entropy）
    - 准确度：被选中的评分平均值
    - 项目重要性指标：∑ div_i * B_scaled_i
    - 以及一个带惩罚项的目标值近似
    """
    recom_num = N * A.shape[1]  # 理论总推荐数
    div = x_result_max.sum(axis=1)  # 每个项目被推荐的次数

    # Entropy-diversity
    entropy_div = 0.0
    for v in div:
        if v > 0:
            p = v / recom_num
            entropy_div += p * math.log(p)
    entropy_div = -entropy_div

    scores_sum = float((A * x_result_max).sum())
    x_sum = int(x_result_max.sum())
    scores_avg = scores_sum / x_sum if x_sum > 0 else 0.0

    # 这里 r_i 只是用来模拟“未达到 Z_i 的惩罚”，在当前简化模型中纯属指标
    r_i = np.maximum(0, Z_value - div)
    max_val = scores_sum - r_i.sum()  # 简化版“精度 - 惩罚”

    item_imp = float((div * B_scaled).sum())

    print(f"评估：")
    print(f"  目标值(max_value 近似)      : {max_val:.4f}")
    print(f"  分布多样性(Entropy)         : {entropy_div:.4f}")
    print(f"  准确度(平均评分 accuracy)   : {scores_avg:.4f}")
    print(f"  项目重要性指标(sum div*B)   : {item_imp:.4f}")
    print(f"  推荐总数(ones)              : {x_sum} （理论应接近 {recom_num}）")

    return dict(
        max_value=max_val,
        entropy=entropy_div,
        accuracy=scores_avg,
        item_importance=item_imp,
        rec_num=x_sum,
        recom_number=recom_num
    )


# =========================================================
# 主程序
# =========================================================
def main():
    print("加载数据中...")
    t0 = time.time()
    A, B = load_data()
    print(f"加载完成，项目数 = {A.shape[0]}, 用户数 = {A.shape[1]}，耗时 {time.time() - t0:.2f}s")

    print("\n=== 第一阶段：代价收益算法计算 Z_i 分布 ===")
    Z_value, min_w, number_die, B_scaled = stage1_cost_benefit_distribution(A, B, N)

    print("\n=== 第二阶段：拉格朗日松弛 + 次梯度 求解推荐矩阵 ===")
    t1 = time.time()
    x_result_max = column_generation_stage2(
        A,
        Z_value,
        N,
        rating_threshold=0.0,   # 你原来就是 0
        max_iter=200,           # 次梯度迭代次数，可调
        step_size=5.0,          # 初始步长，可按数据规模微调
        step_decay=0.95         # 步长衰减系数（0.9~0.99 一般都可以）
    )
    print(f"第二阶段完成，耗时 {time.time() - t1:.2f}s")

    print("\n=== 评估解质量 ===")
    evaluate(A, B_scaled, Z_value, x_result_max, N)

    # 保存推荐矩阵（用户为行、项目为列，你也可以根据需要转置）
    df_res = pd.DataFrame(x_result_max.T)
    outpath = "recommend_result_python_lagrangian.csv"
    df_res.to_csv(outpath, index=False, header=False, encoding='utf-8-sig')
    print(f"\n结果已保存到：{outpath}")


if __name__ == "__main__":
    main()
