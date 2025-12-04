

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.signal import find_peaks

# def load_tum(file_path):
#     """加载TUM格式的轨迹文件，并按时间戳排序。"""
#     try:
#         df = pd.read_csv(file_path, sep=' ', header=None, comment='#')
#         df.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
#         df['timestamp'] = df['timestamp'].astype(float)
#         df.sort_values('timestamp', inplace=True)
#         return df
#     except FileNotFoundError:
#         print(f"错误: 文件未找到 -> {file_path}")
#         return None
#     except Exception as e:
#         print(f"加载文件时出错 {file_path}: {e}")
#         return None

# def plot_trajectories(gt_df, est_df):
#     """使用3D散点图绘制两个轨迹。"""
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(gt_df['x'], gt_df['y'], gt_df['z'], c='green', label='Ground Truth', s=5)
#     ax.scatter(est_df['x'], est_df['y'], est_df['z'], c='red', label='Estimated', s=5)
#     ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
#     ax.set_title('3D Trajectory Comparison'); ax.legend()
#     combined_df = pd.concat([gt_df[['x', 'y', 'z']], est_df[['x', 'y', 'z']]])
#     max_range = np.array([combined_df[c].max()-combined_df[c].min() for c in 'xyz']).max()
#     mid_p = combined_df.mean()
#     ax.set_xlim(mid_p.x - max_range/2, mid_p.x + max_range/2)
#     ax.set_ylim(mid_p.y - max_range/2, mid_p.y + max_range/2)
#     ax.set_zlim(mid_p.z - max_range/2, mid_p.z + max_range/2)




# def plot_advanced_error_graphs(df, along_track_err, cross_track_err, vertical_err):
#     """绘制纵向、横向和垂直误差的2D线图，并在左下角和右下角添加统计图例。"""
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
#     timestamps = df['timestamp']
    
#     ax1.plot(timestamps, along_track_err, 'purple', label='Along-Track Error')
#     ax1.set_ylabel('Error (m)'); ax1.set_title('Along-Track Error (Forward/Backward)'); ax1.grid(True); ax1.legend()
    
#     ax2.plot(timestamps, cross_track_err, 'orange', label='Cross-Track Error')
#     ax2.set_ylabel('Error (m)'); ax2.set_title('Cross-Track Error (Right/Left)'); ax2.grid(True); ax2.legend()

#     ax3.plot(timestamps, vertical_err, 'b-', label='Vertical Error')
#     ax3.set_xlabel('Timestamp (s)'); ax3.set_ylabel('Error (m)'); ax3.set_title('Vertical Error (Up/Down)'); ax3.grid(True); ax3.legend()
    
#     # 调整布局为底部的文本框留出更多空间
#     plt.tight_layout(rect=[0, 0.1, 1, 1]) 
 
#     # --- 核心修改：计算统计数据并创建两个独立的图例框 ---
#     total_points = len(df)
#     if total_points > 0:
#         # --- 第一部分：方向比例统计 ---
#         along_pos_pct = ((along_track_err > 0).sum() / total_points) * 100
#         along_neg_pct = ((along_track_err < 0).sum() / total_points) * 100
#         cross_pos_pct = ((cross_track_err > 0).sum() / total_points) * 100
#         cross_neg_pct = ((cross_track_err < 0).sum() / total_points) * 100
#         vert_pos_pct = ((vertical_err > 0).sum() / total_points) * 100
#         vert_neg_pct = ((vertical_err < 0).sum() / total_points) * 100

#         # 构建方向比例的图例文本
#         direction_legend_text = (
#             f"Error Direction Proportions:\n"
#             f"  Along: Fwd(+) {along_pos_pct:.1f}% | Bwd(-) {along_neg_pct:.1f}%\n"
#             f"  Cross: Right(+) {cross_pos_pct:.1f}% | Left(-) {cross_neg_pct:.1f}%\n"
#             f"  Vert:  Up(+) {vert_pos_pct:.1f}% | Down(-) {vert_neg_pct:.1f}%\n"
#         )

#         # --- 第二部分：绝对值误差统计 ---
#         abs_along = np.abs(along_track_err)
#         abs_cross = np.abs(cross_track_err)
#         abs_vert = np.abs(vertical_err)

#         stats = {
#             "Along-Track": {
#                 "mean": abs_along.mean(), "std": abs_along.std(), "max": abs_along.max(), "min": abs_along.min(),
#                 "rmse": np.sqrt(np.mean(along_track_err**2))
#             },
#             "Cross-Track": {
#                 "mean": abs_cross.mean(), "std": abs_cross.std(), "max": abs_cross.max(), "min": abs_cross.min(),
#                 "rmse": np.sqrt(np.mean(cross_track_err**2))
#             },
#             "Vertical": {
#                 "mean": abs_vert.mean(), "std": abs_vert.std(), "max": abs_vert.max(), "min": abs_vert.min(),
#                 "rmse": np.sqrt(np.mean(vertical_err**2))
#             }
#         }

#         # 构建绝对误差统计的图例文本
#         stats_legend_text = "Absolute Error Statistics (m):\n"
#         for name, data in stats.items():
#             stats_legend_text += (
#                 f"  {name}:\n"
#                 f"    Mean: {data['mean']:.4f} | Std: {data['std']:.4f} | RMSE: {data['rmse']:.4f}\n"
#                 f"    Min: {data['min']:.4f} | Max: {data['max']:.4f}\n"
#             )

#         # --- 第三部分：分别放置两个图例框 ---
#         # 放置方向比例图例在左下角
#         fig.text(0.01, 0.01, direction_legend_text,
#                  ha='left', va='bottom',
#                  bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8),
#                  fontsize=9, family='monospace')

#         # 放置统计数据图例在右下角
#         fig.text(0.99, 0.01, stats_legend_text,
#                  ha='right', va='bottom',
#                  bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8),
#                  fontsize=9, family='monospace')
#     # --- 修改结束 ---




# def plot_trajectory_with_error_heatmap(df, error_values, title):
#     """使用3D散点图绘制轨迹，并用颜色表示误差大小。"""
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     sc = ax.scatter(df['x_est'], df['y_est'], df['z_est'], c=error_values, cmap='viridis', s=10, label='Estimated Trajectory')
#     cbar = fig.colorbar(sc, ax=ax, shrink=0.6); cbar.set_label('Error (m)')
#     ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
#     ax.set_title(title); ax.legend()
#     max_range = np.array([df[f'{c}_est'].max()-df[f'{c}_est'].min() for c in 'xyz']).max()
#     mid_p = df[['x_est', 'y_est', 'z_est']].mean()
#     ax.set_xlim(mid_p.x_est - max_range/2, mid_p.x_est + max_range/2)
#     ax.set_ylim(mid_p.y_est - max_range/2, mid_p.y_est + max_range/2)
#     ax.set_zlim(mid_p.z_est - max_range/2, mid_p.z_est + max_range/2)


# # --- 新增函数：用于绘制带方向的误差热力图 ---
# def plot_trajectory_with_signed_error_heatmap(df, signed_error_values, title):
#     """使用3D散点图绘制轨迹，并用发散颜色图表示带符号的误差。"""
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     # 确定颜色映射的范围，使其关于0对称
#     max_abs_error = np.abs(signed_error_values).max()
    
#     # 使用 'coolwarm' 发散色彩映射: 红色代表正误差, 蓝色代表负误差
#     sc = ax.scatter(df['x_est'], df['y_est'], df['z_est'], 
#                     c=signed_error_values, 
#                     cmap='coolwarm', 
#                     vmin=-max_abs_error, 
#                     vmax=max_abs_error,
#                     s=10, 
#                     label='Estimated Trajectory')
    
#     cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
#     cbar.set_label('Signed Error (m)')
#     ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
#     ax.set_title(title); ax.legend()
    
#     max_range = np.array([df[f'{c}_est'].max()-df[f'{c}_est'].min() for c in 'xyz']).max()
#     mid_p = df[['x_est', 'y_est', 'z_est']].mean()
#     ax.set_xlim(mid_p.x_est - max_range/2, mid_p.x_est + max_range/2)
#     ax.set_ylim(mid_p.y_est - max_range/2, mid_p.y_est + max_range/2)
#     ax.set_zlim(mid_p.z_est - max_range/2, mid_p.z_est + max_range/2)
# # --- 函数结束 ---


# def find_and_print_error_peaks(df, error_series, error_type, num_peaks=5, min_distance_samples=15):
#     """使用信号处理方法查找并打印误差序列中的前N个峰值。"""
#     peaks, properties = find_peaks(error_series, height=error_series.mean(), distance=min_distance_samples)
#     if len(peaks) == 0:
#         print(f"\n--- 未找到显著的 {error_type} 误差峰值 ---")
#         return
#     peak_heights = properties['peak_heights']
#     top_peak_indices_sorted = np.argsort(peak_heights)[-num_peaks:][::-1]
#     print(f"\n--- {error_type} 误差最大的 {min(num_peaks, len(top_peak_indices_sorted))} 个峰值区域 ---")
#     for i in top_peak_indices_sorted:
#         peak_index = peaks[i]
#         peak_row = df.iloc[peak_index]
#         peak_error = error_series.iloc[peak_index]
#         print(f"  时间戳: {peak_row['timestamp']:.6f}, {error_type} 误差: {peak_error:.4f} m")

# # --- 主程序 ---
# if __name__ == "__main__":
#     ground_truth_file = 'outputs/evo_tum/enu_mapping_query/enu_all_query_gt.tum'
#     estimated_file = 'outputs/evo_tum/enu_mapping_query/s_allcams_spatial_colmap_0630_CAM1_0.01_SensorKit_query_centers_tum_filtered.txt'
    
#     gt_df = load_tum(ground_truth_file)
#     est_df = load_tum(estimated_file)

#     if gt_df is None or est_df is None: exit()

#     merged_df = pd.merge_asof(left=est_df, right=gt_df, on='timestamp', direction='nearest', suffixes=('_est', '_gt'))
    
#     dist_3d = np.linalg.norm(merged_df[['x_gt', 'y_gt', 'z_gt']].values - merged_df[['x_est', 'y_est', 'z_est']].values, axis=1)
#     inliers_df = merged_df[dist_3d <= 10.0].copy()
    
#     print(f"总共匹配了 {len(merged_df)} 个点。")
#     print(f"筛选出 {len(inliers_df)} 个误差在 10m 以内的点进行分析。\n")

#     if inliers_df.empty:
#         print("没有找到误差在10米以内的匹配点，无法进行分析。")
#     else:
#         # --- 核心修改：计算带方向的纵向、横向和垂直误差 ---
#         gt_pos = inliers_df[['x_gt', 'y_gt', 'z_gt']].values
#         est_pos = inliers_df[['x_est', 'y_est', 'z_est']].values

#         # 1. 计算GT轨迹的水平前进方向 (忽略Z轴变化)
#         gt_fwd_vec_2d = np.diff(gt_pos[:, :2], axis=0, append=[gt_pos[-1, :2]])
#         gt_fwd_vec_2d_norm = np.linalg.norm(gt_fwd_vec_2d, axis=1, keepdims=True)
#         gt_fwd_vec_2d_norm[gt_fwd_vec_2d_norm == 0] = 1.0
#         gt_fwd_unit_vec_2d = gt_fwd_vec_2d / gt_fwd_vec_2d_norm

#         # 2. 计算水平误差向量
#         error_vec_2d = est_pos[:, :2] - gt_pos[:, :2]

#         # 3. 分解误差
#         # 纵向误差 (Along-Track): 误差向量在前进方向上的投影 (保持不变)
#         along_track_error = np.einsum('ij,ij->i', error_vec_2d, gt_fwd_unit_vec_2d)
        
#         # 横向误差 (Cross-Track): 使用二维叉乘的Z分量来确定方向
#         # cross_product_z = v1.x * v2.y - v1.y * v2.x
#         # 这里 v1 是前进方向, v2 是误差向量
#         # 结果为正表示误差向量在前进方向的“左侧”(逆时针)，我们定义为负
#         # 结果为负表示误差向量在前进方向的“右侧”(顺时针)，我们定义为正
#         cross_product_z = gt_fwd_unit_vec_2d[:, 0] * error_vec_2d[:, 1] - gt_fwd_unit_vec_2d[:, 1] * error_vec_2d[:, 0]
        
#         # 横向误差的大小
#         horizontal_error_sq = np.sum(error_vec_2d**2, axis=1)
#         cross_track_error_magnitude = np.sqrt(np.maximum(0, horizontal_error_sq - along_track_error**2))
        
#         # 结合大小和方向
#         cross_track_error = cross_track_error_magnitude * np.sign(cross_product_z)

#         # 垂直误差 (Vertical): 直接相减，不再取绝对值
#         # 正数表示估计点偏高，负数表示偏低
#         vertical_error = inliers_df['z_est'] - inliers_df['z_gt']
#         # --- 修改结束 ---

#         # --- 将误差数组转换为带索引的 Pandas Series ---
#         along_track_error_s = pd.Series(along_track_error, index=inliers_df.index)
#         cross_track_error_s = pd.Series(cross_track_error, index=inliers_df.index)
#         vertical_error_s = pd.Series(vertical_error.values, index=inliers_df.index)

#         # 打印统计结果 (现在均值和最大值直接计算，不再取abs)
#         print("\n纵向误差 (Along-Track) (m) [+:超前, -:落后]:")
#         print(f"  均值: {along_track_error_s.mean():.4f}")
#         print(f"  最大: {along_track_error_s.max():.4f}")
#         print(f"  最小: {along_track_error_s.min():.4f}")
#         print(f"  RMSE: {np.sqrt(np.mean(along_track_error_s**2)):.4f}")

#         print("\n横向误差 (Cross-Track) (m) [+:右偏, -:左偏]:")
#         print(f"  均值: {cross_track_error_s.mean():.4f}")
#         print(f"  最大: {cross_track_error_s.max():.4f}")
#         print(f"  最小: {cross_track_error_s.min():.4f}")
#         print(f"  RMSE: {np.sqrt(np.mean(cross_track_error_s**2)):.4f}")

#         print("\n垂直误差 (Z-Axis) (m) [+:偏高, -:偏低]:")
#         print(f"  均值: {vertical_error_s.mean():.4f}")
#         print(f"  最大: {vertical_error_s.max():.4f}")
#         print(f"  最小: {vertical_error_s.min():.4f}")
#         print(f"  RMSE: {np.sqrt(np.mean(vertical_error_s**2)):.4f}")

#         # 查找并打印误差峰值 (现在对绝对值查找峰值)
#         find_and_print_error_peaks(inliers_df, np.abs(along_track_error_s), "Along-Track (Magnitude)")
#         find_and_print_error_peaks(inliers_df, np.abs(cross_track_error_s), "Cross-Track (Magnitude)")
#         find_and_print_error_peaks(inliers_df, np.abs(vertical_error_s), "Vertical (Magnitude)")

#         # 绘制新的误差图 (图表本身会显示正负)
#         print("\n正在生成高级误差图...")
#         plot_advanced_error_graphs(inliers_df, along_track_error_s, cross_track_error_s, vertical_error_s)
        
#         # 绘制带误差热力图的轨迹 (热力图仍使用误差大小)
#         print("\n正在生成带误差热力图的轨迹...")
#         plot_trajectory_with_error_heatmap(inliers_df, np.abs(along_track_error_s), 'Trajectory with Along-Track Error Magnitude')
#         plot_trajectory_with_error_heatmap(inliers_df, np.abs(cross_track_error_s), 'Trajectory with Cross-Track Error Magnitude')
#         plot_trajectory_with_error_heatmap(inliers_df, np.abs(vertical_error_s), 'Trajectory with Vertical Error Magnitude')

#         # --- 修改此处：调用新的带方向的热力图函数 ---
#         print("\n正在生成带误差方向热力图的轨迹...")
#         plot_trajectory_with_signed_error_heatmap(inliers_df, along_track_error_s, 'Trajectory with Signed Along-Track Error')
#         plot_trajectory_with_signed_error_heatmap(inliers_df, cross_track_error_s, 'Trajectory with Signed Cross-Track Error')
#         plot_trajectory_with_signed_error_heatmap(inliers_df, vertical_error_s, 'Trajectory with Signed Vertical Error')
#         # --- 修改结束 ---

#     # 绘制原始轨迹图
#     print("\n正在生成轨迹图...")
#     plot_trajectories(gt_df, est_df)

#     plt.show()






















import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks

def load_tum(file_path):
    """加载TUM格式的轨迹文件，并按时间戳排序。"""
    try:
        df = pd.read_csv(file_path, sep=' ', header=None, comment='#')
        df.columns = ['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        # 确保时间戳是浮点数并排序
        df['timestamp'] = df['timestamp'].astype(float)
        df.sort_values('timestamp', inplace=True)
        return df
    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {file_path}")
        return None
    except Exception as e:
        print(f"加载文件时出错 {file_path}: {e}")
        return None

def plot_trajectories(gt_df, est_df):
    """使用3D散点图绘制两个轨迹。"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制地面真值轨迹
    ax.scatter(gt_df['x'], gt_df['y'], gt_df['z'], c='green', label='Ground Truth', s=5)
    
    # 绘制估计轨迹
    ax.scatter(est_df['x'], est_df['y'], est_df['z'], c='red', label='Estimated', s=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory Comparison')
    ax.legend()
    # 尝试设置相等的轴比例
    max_range = np.array([gt_df['x'].max()-gt_df['x'].min(), gt_df['y'].max()-gt_df['y'].min(), gt_df['z'].max()-gt_df['z'].min()]).max()
    mid_x = (gt_df['x'].max()+gt_df['x'].min()) * 0.5
    mid_y = (gt_df['y'].max()+gt_df['y'].min()) * 0.5
    mid_z = (gt_df['z'].max()+gt_df['z'].min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)


def plot_error_graphs(df, horizontal_err, vertical_err):
    """使用2D线图绘制水平和垂直误差。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    timestamps = df['timestamp']
    
    # 绘制水平误差图
    ax1.plot(timestamps, horizontal_err, 'r-', label='Horizontal Error')
    ax1.set_ylabel('Error (m)')
    ax1.set_title('Horizontal Error (XY-Plane) Over Time')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制垂直误差图
    ax2.plot(timestamps, vertical_err, 'b-', label='Vertical Error')
    ax2.set_xlabel('Timestamp (s)')
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Vertical Error (Z-Axis) Over Time')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()

def plot_trajectory_with_error_heatmap(df, error_values, title):
    """
    使用3D散点图绘制轨迹，并用颜色表示误差大小。
    
    Args:
        df (pd.DataFrame): 包含 'x_est', 'y_est', 'z_est' 列的DataFrame。
        error_values (pd.Series): 每个点的误差值。
        title (str): 图表的标题。
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 使用误差值作为颜色，并创建一个颜色映射
    sc = ax.scatter(df['x_est'], df['y_est'], df['z_est'], 
                    c=error_values, cmap='viridis', s=10, label='Estimated Trajectory')
    
    # 添加一个颜色条来解释颜色代表的误差值
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('Error (m)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    # 尝试设置相等的轴比例
    max_range = np.array([df['x_est'].max()-df['x_est'].min(), 
                          df['y_est'].max()-df['y_est'].min(), 
                          df['z_est'].max()-df['z_est'].min()]).max()
    mid_x = (df['x_est'].max()+df['x_est'].min()) * 0.5
    mid_y = (df['y_est'].max()+df['y_est'].min()) * 0.5
    mid_z = (df['z_est'].max()+df['z_est'].min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

# +++ 新增的辅助函数 +++
def find_and_print_error_peaks(df, error_series, error_type, num_peaks=5, min_distance_samples=15):
    """
    使用信号处理方法查找并打印误差序列中的前N个峰值。

    Args:
        df (pd.DataFrame): 包含时间戳等信息的DataFrame。
        error_series (pd.Series): 误差数据。
        error_type (str): 误差的名称，用于打印（例如 "Horizontal"）。
        num_peaks (int): 要查找的峰值数量。
        min_distance_samples (int): 两个峰值之间的最小样本数，用于区分不同的误差事件。
    """
    # 使用 find_peaks 寻找所有满足条件的峰值
    # height: 峰值的最小高度，我们设置为比均值稍高，以过滤掉小的噪声
    # distance: 峰值之间的最小距离，这是区分不同“区域”的关键
    peaks, properties = find_peaks(
        error_series, 
        height=error_series.mean(), 
        distance=min_distance_samples
    )

    if len(peaks) == 0:
        print(f"\n--- 未找到显著的 {error_type} 误差峰值 ---")
        return

    # 从找到的所有峰值中，根据峰值的高度（即误差大小）选出最大的 num_peaks 个
    peak_heights = properties['peak_heights']
    top_peak_indices_sorted = np.argsort(peak_heights)[-num_peaks:][::-1]
    
    print(f"\n--- {error_type} 误差最大的 {min(num_peaks, len(top_peak_indices_sorted))} 个峰值区域 ---")
    
    for i in top_peak_indices_sorted:
        peak_index = peaks[i]
        peak_row = df.iloc[peak_index]
        peak_error = error_series.iloc[peak_index]
        print(f"  时间戳: {peak_row['timestamp']:.6f}, {error_type} 误差: {peak_error:.4f} m")
# +++ 函数结束 +++

# --- 主程序 ---
if __name__ == "__main__":
    # --- 请在这里配置您的文件路径 ---
    ground_truth_file = 'outputs/evo_tum/enu_mapping_query/enu_all_query_gt.tum'
    estimated_file = 'outputs/evo_tum/enu_mapping_query/s_allcams_spatial_colmap_0630_CAM1_0.01_SensorKit_query_centers_tum_filtered.txt'
    
    # 1. 加载数据
    gt_df = load_tum(ground_truth_file)
    est_df = load_tum(estimated_file)

    if gt_df is None or est_df is None:
        exit()

    # 2. 根据最近的时间戳匹配两个轨迹
    # 使用 pandas.merge_asof，它非常适合这种最近邻查找
    merged_df = pd.merge_asof(
        left=est_df, 
        right=gt_df, 
        on='timestamp', 
        direction='nearest', 
        suffixes=('_est', '_gt')
    )

    # 3. 计算3D欧氏距离并筛选
    dist_3d = np.linalg.norm(
        merged_df[['x_gt', 'y_gt', 'z_gt']].values - merged_df[['x_est', 'y_est', 'z_est']].values, 
        axis=1
    )
    
    # 只考虑误差在10米以内的点
    inliers_df = merged_df[dist_3d <= 10.0]
    
    print(f"总共匹配了 {len(merged_df)} 个点。")
    print(f"筛选出 {len(inliers_df)} 个误差在 10m 以内的点进行分析。\n")

    print(f"3D 欧氏距离统计信息:")
    print(f"  最大值: {dist_3d.max():.4f} m")
    print(f"  最小值: {dist_3d.min():.4f} m")
    print(f"  均值: {dist_3d.mean():.4f} m")
    print(f"  标准差: {dist_3d.std():.4f} m")
    print(f"  RMSE: {np.sqrt((dist_3d**2).mean()):.4f} m")

    if inliers_df.empty:
        print("没有找到误差在5米以内的匹配点，无法计算统计数据。")
    else:
        # 4. 计算水平和垂直误差
        horizontal_error = np.sqrt(
            (inliers_df['x_gt'] - inliers_df['x_est'])**2 + 
            (inliers_df['y_gt'] - inliers_df['y_est'])**2
        )
        vertical_error = np.abs(inliers_df['z_gt'] - inliers_df['z_est'])

        # 5. 打印统计结果
        print("水平误差 (m):")
        print(f"  均值: {horizontal_error.mean():.4f}")
        print(f"  最大: {horizontal_error.max():.4f}")
        print(f"  RMSE: {np.sqrt((horizontal_error**2).mean()):.4f}")

        print("\n垂直误差 (m):")
        print(f"  均值: {vertical_error.mean():.4f}")
        print(f"  最大: {vertical_error.max():.4f}")
        print(f"  RMSE: {np.sqrt((vertical_error**2).mean()):.4f}")



        # --- 修改此处：用新的峰值查找函数替换旧的 nlargest 逻辑 ---
        find_and_print_error_peaks(inliers_df, horizontal_error, "Horizontal")
        find_and_print_error_peaks(inliers_df, vertical_error, "Vertical")
        # --- 修改结束 ---

        # 新增：绘制误差图
        print("\n正在生成误差图...")
        plot_error_graphs(inliers_df, horizontal_error, vertical_error)
        
        # 新增：绘制带误差热力图的轨迹
        print("\n正在生成带误差热力图的轨迹...")
        plot_trajectory_with_error_heatmap(inliers_df, horizontal_error, 'Trajectory with Horizontal Error Heatmap')
        plot_trajectory_with_error_heatmap(inliers_df, vertical_error, 'Trajectory with Vertical Error Heatmap')

    # 6. 绘制原始轨迹图
    print("\n正在生成轨迹图...")
    plot_trajectories(gt_df, est_df)

    plt.show()


















# import os
# import struct
# import numpy as np
# import matplotlib.pyplot as plt

# # 数据结构定义
# class Image:
#     def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
#         self.id = id
#         self.qvec = qvec
#         self.tvec = tvec
#         self.camera_id = camera_id
#         self.name = name
#         self.xys = xys
#         self.point3D_ids = point3D_ids

# def qvec2rotmat(qvec):
#     """Convert quaternion to rotation matrix."""
#     w, x, y, z = qvec
#     return np.array([
#         [1 - 2 * y**2 - 2 * z**2,     2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
#         [2 * x * y + 2 * z * w,       1 - 2 * x**2 - 2 * z**2,   2 * y * z - 2 * x * w],
#         [2 * x * z - 2 * y * w,       2 * y * z + 2 * x * w,     1 - 2 * x**2 - 2 * y**2]
#     ])

# def read_images_binary(path_to_model_file):
#     images = {}
#     with open(path_to_model_file, "rb") as f:
#         num_images = struct.unpack("<Q", f.read(8))[0]
#         for _ in range(num_images):
#             image_id = struct.unpack("<I", f.read(4))[0]
#             qvec = struct.unpack("<4d", f.read(32))
#             tvec = struct.unpack("<3d", f.read(24))
#             camera_id = struct.unpack("<I", f.read(4))[0]

#             name = b""
#             while True:
#                 ch = f.read(1)
#                 if ch == b"\x00":
#                     break
#                 name += ch
#             name = name.decode("utf-8")

#             num_points2d = struct.unpack("<Q", f.read(8))[0]
#             f.read(24 * num_points2d)  # skip 2D point data

#             R = qvec2rotmat(qvec)
#             t = np.array(tvec).reshape(3, 1)
#             C = -R.T @ t  # 关键转换：世界坐标中的相机中心位置

#             images[image_id] = {
#                 "qvec": qvec,
#                 "tvec": tvec,
#                 "camera_id": camera_id,
#                 "name": name,
#                 "position": C.flatten()  # 添加这个字段
#             }

#     return images


# def convert_images_to_poses(images):
#     poses = []
#     for image in images.values():
#         qw, qx, qy, qz = image["qvec"]
#         # 使用相机中心位置（已转换）
#         tx, ty, tz = image["position"]
#         timestamp_str = os.path.splitext(image["name"])[0]
#         try:
#             timestamp = int(timestamp_str)
#             poses.append((timestamp, tx, ty, tz, qx, qy, qz, qw))  # 注意：TUM格式中四元数顺序是 x y z w
#         except ValueError:
#             print(f"Skipped image with invalid timestamp: {image['name']}")
#     poses.sort(key=lambda x: x[0])  # 按时间戳排序
#     return poses


# def save_tum_file(poses, out_path):
#     with open(out_path, 'w') as f:
#         for p in poses:
#             f.write(f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

# def plot_trajectory(poses):
#     xs = [p[1] for p in poses]
#     ys = [p[2] for p in poses]
#     zs = [p[3] for p in poses]
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(xs, ys, zs, c='b', marker='o', s=10, label='Camera positions')
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.legend()
#     plt.title("Camera Trajectory (Scatter)")
#     plt.tight_layout()
#     plt.show()



# import numpy as np
# import matplotlib.pyplot as plt

# def qvec2rotmat(qvec):
#     """Convert quaternion to rotation matrix."""
#     w, x, y, z = qvec
#     return np.array([
#         [1 - 2 * y**2 - 2 * z**2,     2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
#         [2 * x * y + 2 * z * w,       1 - 2 * x**2 - 2 * z**2,   2 * y * z - 2 * x * w],
#         [2 * x * z - 2 * y * w,       2 * y * z + 2 * x * w,     1 - 2 * x**2 - 2 * y**2]
#     ])

# def plot_trajectories_comparison_with_orientation(estimated, ground_truth, arrow_length=0.5, step=10):
#     """
#     Plot estimated and ground truth trajectories and camera orientations.
#     arrow_length: length of the orientation arrows
#     step: plot arrows every 'step' points to reduce clutter
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Estimated trajectory
#     xs_est = [p[1] for p in estimated]
#     ys_est = [p[2] for p in estimated]
#     zs_est = [p[3] for p in estimated]
#     ax.plot(xs_est, ys_est, zs_est, 'b-', label='Estimated (COLMAP)')

#     # Ground truth trajectory
#     xs_gt = [p[1] for p in ground_truth]
#     ys_gt = [p[2] for p in ground_truth]
#     zs_gt = [p[3] for p in ground_truth]
#     ax.plot(xs_gt, ys_gt, zs_gt, 'g-', label='Ground Truth')

#     # Draw orientation arrows for estimated poses
#     for i in range(0, len(estimated), step):
#         tx, ty, tz = estimated[i][1], estimated[i][2], estimated[i][3]
#         qx, qy, qz, qw = estimated[i][4], estimated[i][5], estimated[i][6], estimated[i][7]
#         R = qvec2rotmat([qw, qx, qy, qz])
#         # Camera forward direction is the third column of R (camera optical axis)
#         forward = R[:, 2]  
#         ax.quiver(tx, ty, tz, forward[0], forward[1], forward[2], length=arrow_length, color='b', normalize=True)

#     # Draw orientation arrows for ground truth poses
#     for i in range(0, len(ground_truth), step):
#         tx, ty, tz = ground_truth[i][1], ground_truth[i][2], ground_truth[i][3]
#         qx, qy, qz, qw = ground_truth[i][4], ground_truth[i][5], ground_truth[i][6], ground_truth[i][7]
#         R = qvec2rotmat([qw, qx, qy, qz])
#         forward = R[:, 2]
#         ax.quiver(tx, ty, tz, forward[0], forward[1], forward[2], length=arrow_length, color='g', normalize=True)

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.set_title("Trajectory Comparison with Camera Orientation")
#     ax.legend()
#     plt.tight_layout()
#     plt.show()



# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.spatial.transform import Rotation as R
# import numpy as np

# def qvec2rotmat(qvec):
#     """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
#     return R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

# def plot_trajectories_comparison_with_orientation(estimated, ground_truth, arrow_length=0.7, step=10):
#     """
#     Plot estimated and ground truth trajectories and camera orientations.
#     Estimated uses solid RGB axes; Ground Truth uses lighter/different colors for distinction.
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Extract positions
#     pos_est = np.array([[p[1], p[2], p[3]] for p in estimated])
#     pos_gt = np.array([[p[1], p[2], p[3]] for p in ground_truth])

#     # Plot trajectories
#     ax.plot(pos_est[:, 0], pos_est[:, 1], pos_est[:, 2], 'b-', label='Estimated (COLMAP)', linewidth=1.5)
#     ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2], 'g-', label='Ground Truth', linewidth=1.5)

#     # Orientation arrows for Estimated
#     for i in range(0, len(estimated), step):
#         tx, ty, tz = pos_est[i]
#         qx, qy, qz, qw = estimated[i][4:]
#         try:
#             Rmat = qvec2rotmat([qw, qx, qy, qz])
#         except ValueError:
#             continue
#         for axis, color in zip(range(3), ['r', 'g', 'b']):
#             vec = Rmat[:, axis]
#             ax.quiver(tx, ty, tz, vec[0], vec[1], vec[2],
#                       length=arrow_length, color=color, normalize=True, linewidth=3.0)

#     # Orientation arrows for Ground Truth
#     for i in range(0, len(ground_truth), step):
#         tx, ty, tz = pos_gt[i]
#         qx, qy, qz, qw = ground_truth[i][4:]
#         try:
#             Rmat = qvec2rotmat([qw, qx, qy, qz])
#         except ValueError:
#             continue
#         for axis, color in zip(range(3), ['lightcoral', 'lightgreen', 'mediumpurple']):
#             vec = Rmat[:, axis]
#             ax.quiver(tx, ty, tz, vec[0], vec[1], vec[2],
#                       length=arrow_length, color=color, normalize=True, linewidth=1.5)

#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.set_title("Trajectory Comparison with Camera Orientation")
#     ax.legend()
#     plt.tight_layout()
#     plt.show()




# # images_bin_path_local = 'datasets/SensorKit/colmap_seque/sparse/1/images.bin'
# # output_tum_path_local = 'datasets/SensorKit/colmap_seque/sparse/1/local_estimated_from_bin_tum.txt'
# # output_tum_path_local_ = 'datasets/SensorKit/local_estimated_from_bin_tum.txt'


# images_bin_path = 'outputs/SensorKit_hloc/0627_superpoint+lightglue/sfm_superpointmax+lightglue/0.05_BA_CAUCHY_ENU_0626/images.bin'
# output_tum_path = 'outputs/SensorKit_hloc/0627_superpoint+lightglue/sfm_superpointmax+lightglue/0.05_BA_CAUCHY_ENU_0626/estimated_from_bin_tum.txt'
# # output_tum_path_ = 'datasets/SensorKit/align_estimated_from_bin_tum.txt'


# # # 执行流程
# # images_local = read_images_binary(images_bin_path_local)
# # poses_local = convert_images_to_poses(images_local)
# # save_tum_file(poses_local, output_tum_path_local_)
# # plot_trajectory(poses_local)


# images = read_images_binary(images_bin_path)
# poses = convert_images_to_poses(images)
# save_tum_file(poses, output_tum_path)
# plot_trajectory(poses)



# def parse_gt_file(filepath, timestamps_to_keep):
#     poses = []
#     with open(filepath, 'r') as f:
#         lines = f.readlines()

#     for line_idx, line in enumerate(lines):
#         # Skip header line if present (optional, based on your actual file)
#         if line_idx == 0 and "image_timestamp_ns" in line: # Basic check for a header
#             continue

#         tokens = line.strip().split(',') # Split by comma for CSV-like format
#         if len(tokens) != 11:
#             print(f"Warning: Skipping line {line_idx+1} due to incorrect number of columns: {len(tokens)} found, 11 expected.")
#             continue

#         try:
#             # Use image_timestamp_ns (first column) as the primary timestamp
#             timestamp_ns_str = tokens[0]
#             timestamp_ns = int(timestamp_ns_str)

#             if timestamp_ns not in timestamps_to_keep:
#                 continue

#             # pos_x_ecef, pos_y_ecef, pos_z_ecef
#             tx = float(tokens[3])
#             ty = float(tokens[4])
#             tz = float(tokens[5])

#             # quat_w_c2e, quat_x_c2e, quat_y_c2e, quat_z_c2e
#             qw_c2e = float(tokens[6])
#             qx_c2e = float(tokens[7])
#             qy_c2e = float(tokens[8])
#             qz_c2e = float(tokens[9])
            
#             # Store in TUM order for quaternion: qx, qy, qz, qw
#             # This matches the order used later by plot_trajectories_comparison_with_orientation
#             # where qx, qy, qz, qw = estimated[i][4:]
#             # and then qvec2rotmat([qw, qx, qy, qz]) is called.
#             poses.append((timestamp_ns, tx, ty, tz, qx_c2e, qy_c2e, qz_c2e, qw_c2e))

#         except ValueError as e:
#             print(f"Warning: Skipping line {line_idx+1} due to value error: {e} - Line content: {line.strip()}")
#             continue
            
#     poses.sort(key=lambda x: x[0])
#     return poses

# gt_txt_path = 'datasets/SensorKit/mapping_camera_ecef_poses_with_quaternion.txt'
# output_gt_tum = 'datasets/SensorKit/tum_mapping_camera_ecef_poses_with_quaternion.txt'
# timestamps_est = set(p[0] for p in poses)  # 来自上一步的estimated pose

# gt_poses = parse_gt_file(gt_txt_path, timestamps_est)
# save_tum_file(gt_poses, output_gt_tum)



# plot_trajectories_comparison_with_orientation(poses, gt_poses, arrow_length=20, step=100)












# import os
# import struct
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# from collections import defaultdict

# # 数据结构定义
# class Image:
#     def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
#         self.id = id
#         self.qvec = qvec
#         self.tvec = tvec
#         self.camera_id = camera_id
#         self.name = name
#         self.xys = xys
#         self.point3D_ids = point3D_ids

# def qvec2rotmat(qvec):
#     """Convert quaternion to rotation matrix."""
#     w, x, y, z = qvec
#     return np.array([
#         [1 - 2 * y**2 - 2 * z**2,     2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
#         [2 * x * y + 2 * z * w,       1 - 2 * x**2 - 2 * z**2,   2 * y * z - 2 * x * w],
#         [2 * x * z - 2 * y * w,       2 * y * z + 2 * x * w,     1 - 2 * x**2 - 2 * y**2]
#     ])

# def read_images_binary(path_to_model_file):
#     images = {}
#     with open(path_to_model_file, "rb") as f:
#         num_images = struct.unpack("<Q", f.read(8))[0]
#         for _ in range(num_images):
#             image_id = struct.unpack("<I", f.read(4))[0]
#             qvec = struct.unpack("<4d", f.read(32))
#             tvec = struct.unpack("<3d", f.read(24))
#             camera_id = struct.unpack("<I", f.read(4))[0]

#             name = b""
#             while True:
#                 ch = f.read(1)
#                 if ch == b"\x00":
#                     break
#                 name += ch
#             name = name.decode("utf-8")

#             num_points2d = struct.unpack("<Q", f.read(8))[0]
#             f.read(24 * num_points2d)  # skip 2D point data

#             R = qvec2rotmat(qvec)
#             t = np.array(tvec).reshape(3, 1)
#             C = -R.T @ t  # 关键转换：世界坐标中的相机中心位置

#             images[image_id] = {
#                 "qvec": qvec,
#                 "tvec": tvec,
#                 "camera_id": camera_id,
#                 "name": name,
#                 "position": C.flatten()  # 添加这个字段
#             }

#     return images


# # # 修复方向朝向错误
# # def convert_images_to_poses(images):
# #     poses = []
# #     for image in images.values():
# #         qw, qx, qy, qz = image["qvec"]

# #         # 使用共轭四元数表示从相机坐标系到世界坐标系的旋转
# #         qx, qy, qz, qw = -qx, -qy, -qz, qw

# #         tx, ty, tz = image["position"]
# #         timestamp_str = os.path.splitext(image["name"])[0]
# #         try:
# #             timestamp = int(timestamp_str)
# #             poses.append((timestamp, tx, ty, tz, qx, qy, qz, qw))
# #         except ValueError:
# #             print(f"Skipped image with invalid timestamp: {image['name']}")
# #     poses.sort(key=lambda x: x[0])
# #     return poses


# # def save_tum_file(poses, out_path):
# #     with open(out_path, 'w') as f:
# #         for p in poses:
# #             f.write(f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")



# # 修复方向朝向错误
# def convert_images_to_poses(images):
#     poses = []
#     for image in images.values():
#         qw, qx, qy, qz = image["qvec"]

#         # 使用共轭四元数表示从相机坐标系到世界坐标系的旋转
#         qx, qy, qz, qw = -qx, -qy, -qz, qw

#         tx, ty, tz = image["position"]
#         full_image_name = image["name"] # e.g., "cam6/1740554624210021238.jpg"
        
#         # 从完整图像名中提取时间戳用于排序
#         try:
#             # 假设格式为 "cam_id/timestamp.jpg"
#             timestamp_str = os.path.splitext(full_image_name.split('/')[-1])[0]
#             timestamp = int(timestamp_str)
#             # 在元组中同时保存完整图像名和时间戳
#             poses.append((timestamp, full_image_name, tx, ty, tz, qx, qy, qz, qw))
#         except (ValueError, IndexError):
#             print(f"Skipped image with invalid name format: {full_image_name}")
            
#     # 按时间戳排序
#     poses.sort(key=lambda x: x[0])
#     return poses


# def save_tum_file(poses, out_path):
#     with open(out_path, 'w') as f:
#         # p 的格式是 (timestamp, full_image_name, tx, ty, tz, qx, qy, qz, qw)
#         for p in poses:
#             # 将 full_image_name 作为第一列写入
#             f.write(f"{p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]} {p[8]}\n")



# images_bin_path = 'outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.01/sfm_superpointmax+lightglue/images.bin'
# output_tum_path = 'outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.01/sfm_superpointmax+lightglue/estimated_from_bin_tum.txt'


# images = read_images_binary(images_bin_path)
# poses = convert_images_to_poses(images)
# save_tum_file(poses, output_tum_path)




# # def generate_tum_files_by_camera(poses_path, mapping_path, output_dir="tum_by_camera"):
# #     """
# #     根据相机ID将一个姿态文件分割成多个TUM格式的轨迹文件。

# #     Args:
# #         poses_path (str): 包含姿态数据的文件路径。
# #                           格式: image_name 已经是 tx ty tz qx qy qz qw 
# #         mapping_path (str): 包含 image_id 到 camera_id 映射的CSV文件路径。
# #                             格式: image_id,camera
# #         output_dir (str): 用于存放生成的TUM文件的目录。
# #     """
# #     # --- 1. 加载图像到相机的映射 ---
# #     if not os.path.exists(mapping_path):
# #         print(f"错误：找不到映射文件 '{mapping_path}'")
# #         return

# #     try:
# #         # 使用 pandas 读取映射文件，分隔符为空白符，可以处理空格或tab
# #         mapping_df = pd.read_csv(mapping_path, sep=',')
# #         # 创建一个字典以便快速查找，同时去除文件名中的 .jpg 后缀
# #         image_to_camera_map = {
# #             os.path.splitext(row['image_id'])[0]: row['camera']
# #             for _, row in mapping_df.iterrows()
# #         }
# #         print(f"成功加载了 {len(image_to_camera_map)} 条图像到相机的映射。")
# #     except Exception as e:
# #         print(f"读取映射文件时出错: {e}")
# #         return

# #     # --- 2. 读取并处理姿态数据 ---
# #     if not os.path.exists(poses_path):
# #         print(f"错误：找不到姿态文件 '{poses_path}'")
# #         return

# #     # 使用 defaultdict 可以方便地按相机ID对姿态行进行分组
# #     camera_trajectories = defaultdict(list)

# #     with open(poses_path, 'r') as f_in:
# #         for line in f_in:
# #             # 跳过空行或注释行
# #             if not line.strip() or line.strip().startswith('#'):
# #                 continue
            
# #             parts = line.strip().split()
# #             if len(parts) != 8:
# #                 print(f"警告：跳过格式不正确的行 -> {line.strip()}")
# #                 continue

# #             image_name = parts[0]
            
# #             # 查找对应的相机ID
# #             camera_id = image_to_camera_map.get(image_name)

# #             if camera_id:
# #                 # 提取数据
# #                 timestamp = parts[0]
# #                 tx, ty, tz = parts[1], parts[2], parts[3]
# #                 qx, qy, qz, qw = parts[4], parts[5], parts[6], parts[7]

# #                 try:
# #                     timestamp_s = float(timestamp) / 1e9
# #                 except ValueError:
# #                     print(f"警告：无法将时间戳 '{timestamp}' 转换为数字，跳过此行。")
# #                     continue

# #                 # **关键转换**: 使用秒为单位的时间戳，并调整四元数顺序以符合TUM格式 (qx qy qz qw)
# #                 tum_line = f"{timestamp_s:.6f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n"
                
# #                 # 将格式化后的行添加到对应相机的列表中
# #                 camera_trajectories[camera_id].append(tum_line)
# #             else:
# #                 print(f"警告：在映射文件中找不到图像 '{image_name}' 对应的相机ID。")

# #     # --- 3. 写入到独立的TUM文件 ---
# #     if not camera_trajectories:
# #         print("没有找到任何可以处理的有效姿态数据。")
# #         return

# #     # 创建输出目录（如果不存在）
# #     os.makedirs(output_dir, exist_ok=True)

# #     for cam_id, lines in camera_trajectories.items():
# #         # 按时间戳对每个相机的轨迹进行排序
# #         lines.sort(key=lambda x: float(x.split()[0]))
        
# #         output_path = os.path.join(output_dir, f"estimated_mapping_{cam_id}.tum")
# #         with open(output_path, 'w') as f_out:
# #             # 写入TUM文件头
# #             f_out.write("# timestamp tx ty tz qx qy qz qw\n")
# #             f_out.writelines(lines)
        
# #         print(f"成功为 '{cam_id}' 创建了轨迹文件，包含 {len(lines)} 个位姿 -> {output_path}")


# # # --- 主程序入口 ---
# # if __name__ == "__main__":
# #     # --- 请在这里配置您的文件路径 ---
    
# #     # 1. 您的姿态数据文件 (image_name tx ty tz qx qx qy qz)
# #     poses_data_file = "outputs/SensorKit_hloc/test_t_aachen+lightglue/sfm_superpointmax+lightglue/estimated_from_bin_tum.txt"
    
# #     # 2. 您的映射文件 (image_id camera)
# #     image_camera_mapping_file = "datasets/SensorKit/mapping_image_matched.csv"
    
# #     # 3. 输出TUM文件的文件夹名称
# #     output_directory = "outputs/evo_tum/enu_mapping_query/cam1_test_t_aachen+lightglue_estimated_mapping"

# #     # 调用主函数执行转换
# #     generate_tum_files_by_camera(poses_data_file, image_camera_mapping_file, output_directory)






# def generate_tum_files_by_camera(poses_path, output_dir="tum_by_camera"):
#     """
#     根据姿态文件中的图像名直接提取相机ID，并将姿态分割成多个TUM格式的轨迹文件。

#     Args:
#         poses_path (str): 包含姿态数据的文件路径。
#                           格式: cam_id/image_name.jpg tx ty tz qx qy qz qw
#         output_dir (str): 用于存放生成的TUM文件的目录。
#     """
#     # --- 1. 读取并处理姿态数据 ---
#     if not os.path.exists(poses_path):
#         print(f"错误：找不到姿态文件 '{poses_path}'")
#         return

#     # 使用 defaultdict 可以方便地按相机ID对姿态行进行分组
#     camera_trajectories = defaultdict(list)

#     with open(poses_path, 'r') as f_in:
#         for line in f_in:
#             # 跳过空行或注释行
#             if not line.strip() or line.strip().startswith('#'):
#                 continue
            
#             parts = line.strip().split()
#             if len(parts) != 8:
#                 print(f"警告：跳过格式不正确的行 -> {line.strip()}")
#                 continue

#             full_image_name = parts[0]
            
#             # --- 核心修改：直接从图像名解析相机ID ---
#             try:
#                 # 假设格式为 "cam_id/image_name.jpg"
#                 camera_id, image_name = full_image_name.split('/', 1)
#             except ValueError:
#                 print(f"警告：无法从 '{full_image_name}' 中解析相机ID，跳过此行。")
#                 continue
#             # --- 修改结束 ---

#             # 提取数据
#             timestamp_str = os.path.splitext(image_name)[0]
#             tx, ty, tz = parts[1], parts[2], parts[3]
#             qx, qy, qz, qw = parts[4], parts[5], parts[6], parts[7]

#             try:
#                 # 假设时间戳是纳秒
#                 timestamp_s = float(timestamp_str) / 1e9
#             except ValueError:
#                 print(f"警告：无法将时间戳 '{timestamp_str}' 转换为数字，跳过此行。")
#                 continue

#             # 格式化为TUM格式的行
#             tum_line = f"{timestamp_s:.6f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n"
            
#             # 将格式化后的行添加到对应相机的列表中
#             camera_trajectories[camera_id].append(tum_line)

#     # --- 2. 写入到独立的TUM文件 ---
#     if not camera_trajectories:
#         print("没有找到任何可以处理的有效姿态数据。")
#         return

#     # 创建输出目录（如果不存在）
#     os.makedirs(output_dir, exist_ok=True)

#     for cam_id, lines in camera_trajectories.items():
#         # 按时间戳对每个相机的轨迹进行排序
#         lines.sort(key=lambda x: float(x.split()[0]))
        
#         output_path = os.path.join(output_dir, f"estimated_mapping_{cam_id}.tum")
#         with open(output_path, 'w') as f_out:
#             # 写入TUM文件头
#             f_out.write("# timestamp tx ty tz qx qy qz qw\n")
#             f_out.writelines(lines)
        
#         print(f"成功为 '{cam_id}' 创建了轨迹文件，包含 {len(lines)} 个位姿 -> {output_path}")


# # --- 主程序入口 ---
# if __name__ == "__main__":
#     # --- 请在这里配置您的文件路径 ---
    
#     # 1. 您的姿态数据文件 (cam_id/image_name.jpg tx ty tz qx qy qz qw)
#     # 注意：这里的 image_name 已经是 tx ty tz qx qy qz qw 格式，
#     # 但第一列是 cam_id/image_name.jpg
#     poses_data_file = "outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.01/sfm_superpointmax+lightglue/estimated_from_bin_tum.txt"
    
#     # 2. 输出TUM文件的文件夹名称
#     output_directory = "outputs/evo_tum/enu_mapping_query/0.01_cam_split_test_t_aachen+lightglue"

#     # 调用主函数执行转换 (不再需要 mapping 文件)
#     generate_tum_files_by_camera(poses_data_file, output_directory)





























# import numpy as np

# def remove_outliers(input_path, output_path, threshold=10.0):
#     data = []
#     with open(input_path) as f:
#         for line in f:
#             if line.startswith("#"):
#                 continue
#             parts = line.strip().split()
#             if len(parts) != 8:
#                 continue
#             timestamp_ns = float(parts[0])
#             timestamp_s = timestamp_ns / 1e9  # Convert nanoseconds to seconds
#             tx, ty, tz = map(float, parts[1:4])
#             quat = list(map(float, parts[4:]))
#             data.append([timestamp_s, tx, ty, tz] + quat)

#     data = np.array(data, dtype=float)
#     positions = data[:, 1:4]

#     center = np.mean(positions, axis=0)
#     dists = np.linalg.norm(positions - center, axis=1)

#     filtered = data[dists < threshold]

#     with open(output_path, "w") as f_out:
#         for row in filtered:
#             # Ensure timestamp (row[0]) is formatted correctly, e.g., with 6 decimal places for seconds
#             line = f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} " + " ".join(f"{v:.6f}" for v in row[4:]) + "\n"
#             f_out.write(line)

#     print(f"Filtered file saved to {output_path}, removed {len(data) - len(filtered)} outliers.")


# remove_outliers("outputs/evo_tum/local_mapping_query/timeprocess_SensorKit_query_centers_tum.txt",
#                 "outputs/evo_tum/local_mapping_query/s_timeprocess_SensorKit_query_centers_tum_filtered.txt",
#                 threshold=5.0)  # threshold自己调






# ########### ns TUM s
# import numpy as np

# def convert_timestamp_ns_to_s(input_path, output_path):
#     """
#     Reads a TUM-like file, converts the first column (timestamp) from nanoseconds to seconds,
#     and saves the modified data to a new file.
#     """
#     output_lines = []
#     with open(input_path, 'r') as f_in:
#         for line in f_in:
#             if line.startswith("#"): # Preserve comment lines
#                 output_lines.append(line)
#                 continue
            
#             parts = line.strip().split()
#             if len(parts) == 8: # Expecting timestamp tx ty tz qx qy qz qw
#                 try:
#                     timestamp_ns = float(parts[0])
#                     timestamp_s = timestamp_ns / 1e9  # Convert nanoseconds to seconds
                    
#                     # Reconstruct the line with the new timestamp
#                     # and format all numbers to .6f
#                     new_line_parts = [f"{timestamp_s:.6f}"] + [f"{float(p):.6f}" for p in parts[1:]]
#                     output_lines.append(" ".join(new_line_parts) + "\n")
#                 except ValueError:
#                     print(f"Warning: Could not parse numbers in line, skipping: {line.strip()}")
#                     output_lines.append(line) # Append original line if parsing fails
#             else:
#                 # If the line doesn't have 8 parts, append it as is (e.g., empty lines or malformed)
#                 output_lines.append(line)

#     with open(output_path, "w") as f_out:
#         for out_line in output_lines:
#             f_out.write(out_line)
    
#     print(f"Converted timestamps and saved to {output_path}")


# input_file_path = "outputs/evo_tum/enu_mapping_query/0.05_BA_CAUCHY_ENU_0626_CAM2_SensorKit_query_centers_tum.txt"
# output_file_path = "outputs/evo_tum/enu_mapping_query/s_0.05_BA_CAUCHY_ENU_0626_CAM2_SensorKit_query_centers_tum.txt" # New output file name

# import os
# if os.path.exists(input_file_path):
#     convert_timestamp_ns_to_s(input_file_path, output_file_path)
# else:
#     print(f"Input file not found: {input_file_path}")











# import numpy as np
# from bisect import bisect_left

# def load_tum_file(file_path):
#     """
#     Load TUM trajectory into a sorted list: [[timestamp, tx, ty, tz, qx, qy, qz, qw]]
#     """
#     data = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             if line.strip() and not line.startswith("#"):
#                 parts = line.strip().split()
#                 if len(parts) == 8:
#                     timestamp = float(parts[0])
#                     values = list(map(float, parts[1:]))
#                     data.append([timestamp] + values)
#     data.sort()  # ensure sorted by timestamp
#     return data

# def find_nearest_index(timestamps, target_time):
#     """
#     Find the index of the timestamp in 'timestamps' that is closest to 'target_time'
#     """
#     idx = bisect_left(timestamps, target_time)
#     if idx == 0:
#         return 0
#     if idx == len(timestamps):
#         return len(timestamps) - 1
#     before = timestamps[idx - 1]
#     after = timestamps[idx]
#     return idx - 1 if abs(before - target_time) < abs(after - target_time) else idx

# def match_and_filter_nearest(est_data, gt_data, threshold=10.0):
#     """
#     Match est to gt using nearest timestamp and filter based on 3D distance
#     """
#     gt_timestamps = [row[0] for row in gt_data]
#     gt_dict = {row[0]: row[1:4] for row in gt_data}

#     filtered = []
#     for est_row in est_data:
#         est_time = est_row[0]
#         est_pos = np.array(est_row[1:4])

#         nearest_idx = find_nearest_index(gt_timestamps, est_time)
#         gt_time = gt_timestamps[nearest_idx]
#         gt_pos = np.array(gt_dict[gt_time])

#         dist = np.linalg.norm(est_pos - gt_pos)
#         if dist <= threshold:
#             filtered.append(est_row)
#         else:
#             print(f"Frame at {est_time:.6f} rejected (distance = {dist:.3f}m to nearest gt @ {gt_time:.6f})")

#     return np.array(filtered)

# def save_tum_trajectory(file_path, data):
#     with open(file_path, 'w') as f:
#         for row in data:
#             f.write("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*row))

# if __name__ == "__main__":
#     est_file = "outputs/evo_tum/enu_mapping_query/s_0.05_BA_CAUCHY_ENU_SensorKit_query_centers_tum.txt"
#     gt_file = "outputs/evo_tum/enu_mapping_query/enu_all_query_gt.tum"
#     output_file = "outputs/evo_tum/enu_mapping_query/s_0.05_BA_CAUCHY_ENU_SensorKit_query_centers_tum_filtered.txt"
#     threshold = 10.0  # meters # 10.0

#     print("Loading estimated trajectory...")
#     est_data = load_tum_file(est_file)

#     print("Loading ground truth trajectory...")
#     gt_data = load_tum_file(gt_file)

#     print("Filtering using nearest time and distance threshold...")
#     filtered_data = match_and_filter_nearest(est_data, gt_data, threshold)

#     print(f"Saving filtered trajectory to {output_file}...")
#     save_tum_trajectory(output_file, filtered_data)

#     print(f"Done. Original: {len(est_data)} frames, Filtered: {len(filtered_data)} frames")








# import matplotlib.pyplot as plt
# import numpy as np

# def load_tum(path):
#     poses = []
#     with open(path) as f:
#         for line in f:
#             if line.startswith("#") or len(line.strip().split()) != 8:
#                 continue
#             parts = list(map(float, line.strip().split()))
#             poses.append(parts[1:4])  # tx, ty, tz
#     return np.array(poses)

# est = load_tum("outputs/evo_tum/local_mapping_query/local_estimated_from_bin_tum.txt")
# gt = load_tum("outputs/evo_tum/local_mapping_query/SensorKit_query_centers_tum_filtered.txt")

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='blue', label='Ground Truth', s=0.5)
# ax.scatter(est[:, 0], est[:, 1], est[:, 2], c='red', label='Estimated', s=0.5)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# plt.show()





# import csv

# input_file = "datasets/SensorKit/query_camera_ecef_poses_with_quaternion.txt"  # 你的原始txt/csv文件路径
# output_file = "outputs/evo_tum/ecef_mapping_query/tum_query_camera_ecef_poses_gt.txt"  # 输出TUM格式文件路径

# with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
#     reader = csv.DictReader(f_in)
#     for row in reader:
#         timestamp = row['image_timestamp_ns']  # 直接保留原始纳秒时间戳（整数，不转换）
#         pos_x = row['pos_x_ecef']
#         pos_y = row['pos_y_ecef']
#         pos_z = row['pos_z_ecef']
#         quat_w = row['quat_w_c2e']
#         quat_x = row['quat_x_c2e']
#         quat_y = row['quat_y_c2e']
#         quat_z = row['quat_z_c2e']

#         line = f"{timestamp} {pos_x} {pos_y} {pos_z} {quat_w} {quat_x} {quat_y} {quat_z}\n"
#         f_out.write(line)


















# import matplotlib.pyplot as plt
# import pandas as pd
# import io
# import numpy as np # 新增导入 numpy

# with open("datasets/SensorKit/new_colmap_spatial_intrinsics/model_alignment_stats.txt", "r") as f:
#     data_string = f.read()


# # 定义列名
# column_names = ['Model_Index', 'Mean_Alignment_Error', 'Median_Alignment_Error', 'Num_Registered_Images']

# # 使用 pandas 读取数据
# df = pd.read_csv(io.StringIO(data_string), comment='#', sep=r'\s+', names=column_names)

# # 按 'Num_Registered_Images' 排序，以便绘图时X轴有序（对于柱状图，这会影响类别的顺序）
# df = df.sort_values(by='Num_Registered_Images')

# # 提取数据列
# # 将 Num_Registered_Images 转换为字符串类型，以便在X轴上作为离散类别处理
# # 否则，如果它是数字，matplotlib 可能会尝试将其视为连续值
# num_images_categories = df['Num_Registered_Images'].astype(str) 
# mean_error = df['Mean_Alignment_Error']
# median_error = df['Median_Alignment_Error']

# # 创建图表
# fig, ax1 = plt.subplots(figsize=(15, 8)) # 调整图形大小以容纳更多柱子

# # 设置柱状图的位置
# x = np.arange(len(num_images_categories))  # X轴上的标签位置
# width = 0.35  # 柱子的宽度

# # 绘制 Mean Alignment Error 的柱子
# rects1 = ax1.bar(x - width/2, mean_error, width, label='Mean Alignment Error', color='tab:blue')

# # 创建第二个Y轴共享同一个X轴，用于 Median Alignment Error
# ax2 = ax1.twinx()
# # 绘制 Median Alignment Error 的柱子
# rects2 = ax2.bar(x + width/2, median_error, width, label='Median Alignment Error', color='tab:red')

# # 添加标签、标题和自定义X轴刻度标签等。
# ax1.set_xlabel('Number of Registered Images')
# ax1.set_ylabel('Mean Alignment Error', color='tab:blue')
# ax2.set_ylabel('Median Alignment Error', color='tab:red')
# ax1.set_title('Alignment Error vs. Number of Registered Images')
# ax1.set_xticks(x)
# ax1.set_xticklabels(num_images_categories, rotation=45, ha="right") # 旋转X轴标签以防重叠
# ax1.tick_params(axis='y', labelcolor='tab:blue')
# ax2.tick_params(axis='y', labelcolor='tab:red')

# # 添加图例
# # 合并图例句柄和标签
# handles1, labels1 = ax1.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)


# # 自动调整子图参数，以便它们很好地适应图形区域
# fig.tight_layout(rect=[0, 0, 1, 0.9]) # 调整 rect 为图例和旋转的X轴标签留出空间

# # 显示图表
# plt.show()

# # 如果需要保存图表
# # output_plot_path = "alignment_error_vs_num_images_bar.png"
# # plt.savefig(output_plot_path)
# # print(f"Plot saved to {output_plot_path}")
