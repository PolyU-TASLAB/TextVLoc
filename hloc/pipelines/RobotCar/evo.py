# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation as R
# import csv # 导入 CSV 模块
# import os  # 导入 os 模块以提取文件名

# def read_gt_csv_pose_file(filepath):
#     """
#     读取包含 ECEF 位姿的 CSV 真值文件。
#     CSV 列应包含: image_path, quat_w_c2e, quat_x_c2e, quat_y_c2e, quat_z_c2e, 
#                      pos_x_ecef, pos_y_ecef, pos_z_ecef,
#                      image_timestamp_ns, camera_id (可选)
#     """
#     poses = {}
#     # 更新预期表头以包含时间戳和相机ID
#     expected_headers_minimal = ['image_path', 'quat_w_c2e', 'quat_x_c2e', 'quat_y_c2e', 'quat_z_c2e',
#                                 'pos_x_ecef', 'pos_y_ecef', 'pos_z_ecef']
#     optional_headers = ['image_timestamp_ns', 'camera_id'] # 这些是可选的
    
#     try:
#         with open(filepath, 'r', newline='') as f:
#             reader = csv.DictReader(f)
            
#             actual_fieldnames = reader.fieldnames
#             if actual_fieldnames is None:
#                 print(f"错误: CSV 文件 {filepath} 为空或无法读取表头。")
#                 return poses

#             if not all(header in actual_fieldnames for header in expected_headers_minimal):
#                 print(f"错误: CSV 文件 {filepath} 的表头缺少必要的列。")
#                 print(f"预期至少包含: {expected_headers_minimal}")
#                 print(f"实际表头: {actual_fieldnames}")
#                 return poses

#             has_timestamp = 'image_timestamp_ns' in actual_fieldnames
#             has_camera_id = 'camera_id' in actual_fieldnames

#             for row in reader:
#                 try:
#                     image_full_path = row['image_path']
#                     name = os.path.basename(image_full_path)
                    
#                     q_val = np.array([
#                         float(row['quat_w_c2e']), 
#                         float(row['quat_x_c2e']), 
#                         float(row['quat_y_c2e']), 
#                         float(row['quat_z_c2e'])
#                     ])
#                     pos_ecef_val = np.array([
#                         float(row['pos_x_ecef']), 
#                         float(row['pos_y_ecef']), 
#                         float(row['pos_z_ecef'])
#                     ])
                    
#                     pose_data = {'q': q_val, 'pos_ecef': pos_ecef_val}
                    
#                     if has_timestamp:
#                         pose_data['timestamp_ns'] = row.get('image_timestamp_ns', '') # 使用 get 以防万一
#                     if has_camera_id:
#                         pose_data['camera_id'] = row.get('camera_id', '')

#                     poses[name] = pose_data
#                 except ValueError as e:
#                     print(f"警告: 解析 CSV 行时出错: {row} - {e}")
#                     continue
#                 except KeyError as e:
#                     print(f"警告: CSV 行中缺少预期的键: {e} 在行 {row}")
#                     continue
#     except FileNotFoundError:
#         print(f"错误: 找不到真值文件 {filepath}")
#     except Exception as e:
#         print(f"读取或解析 CSV 文件 {filepath} 时发生错误: {e}")
#     return poses

# def read_pred_txt_pose_file(filepath):
#     """
#     读取预测的 TXT 位姿文件。
#     文件格式: image_name qw qx qy qz pos_x pos_y pos_z (pos_x, pos_y, pos_z 是 ECEF)
#     """
#     poses = {}
#     try:
#         with open(filepath, 'r') as f:
#             for line in f:
#                 if line.startswith("#"):
#                     continue
#                 parts = line.strip().split()
#                 if len(parts) != 8:
#                     print(f"警告 (预测文件): 跳过格式不正确的行: {line.strip()}")
#                     continue
                
#                 name = parts[0]
#                 try:
#                     q = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
#                     pos_ecef = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
#                     poses[name] = {'q': q, 'pos_ecef': pos_ecef}
#                 except ValueError as e:
#                     print(f"警告 (预测文件): 解析行时出错: {line.strip()} - {e}")
#                     continue
#     except FileNotFoundError:
#         print(f"错误: 找不到预测文件 {filepath}")
#     except Exception as e:
#         print(f"读取或解析预测文件 {filepath} 时发生错误: {e}")
#     return poses


# def ecef_to_enu(ecef_points, ref_ecef):
#     """
#     将 ECEF 坐标点（或点列表）转换为相对于参考 ECEF 点的本地 ENU 坐标。
#     这是一个简化的转换，假设地球是球体，并且不考虑精确的大地水准面。
#     对于高精度应用，请使用专用地理库。

#     Args:
#         ecef_points (np.ndarray): Shape (3,) 或 (N, 3) 的 ECEF 坐标。
#         ref_ecef (np.ndarray): Shape (3,) 的参考点 ECEF 坐标。

#     Returns:
#         np.ndarray: Shape (3,) 或 (N, 3) 的 ENU 坐标。
#     """
#     # WGS84 参数
#     a = 6378137.0  # 地球半长轴
#     e_sq = 6.69437999014e-3  # 第一偏心率的平方

#     ref_x, ref_y, ref_z = ref_ecef
#     p_ref = np.sqrt(ref_x**2 + ref_y**2)
    
#     # 计算参考点的经纬度
#     lon_ref = np.arctan2(ref_y, ref_x)
    
#     # 迭代计算大地纬度 (更精确的方法)
#     lat_ref = np.arctan2(ref_z, p_ref * (1 - e_sq)) # 初始猜测
#     # N_ref_prev = a / np.sqrt(1 - e_sq * np.sin(lat_ref)**2) # 初始化 N_ref
#     for _ in range(10): # 迭代几次以提高精度
#         sin_lat_ref = np.sin(lat_ref)
#         N_ref = a / np.sqrt(1 - e_sq * sin_lat_ref**2)
#         h_ref = p_ref / np.cos(lat_ref) - N_ref # 高度（虽然这里我们主要需要纬度）
#         lat_ref_new = np.arctan2(ref_z, p_ref * (1 - e_sq * N_ref / (N_ref + h_ref)))
#         if np.abs(lat_ref_new - lat_ref) < 1e-9: # 收敛条件
#             lat_ref = lat_ref_new
#             break
#         lat_ref = lat_ref_new
#     # N_ref = a / np.sqrt(1 - e_sq * np.sin(lat_ref)**2) # 最终 N_ref


#     # 构建从 ECEF 到 ENU 的旋转矩阵
#     sin_lat_ref = np.sin(lat_ref)
#     cos_lat_ref = np.cos(lat_ref)
#     sin_lon_ref = np.sin(lon_ref)
#     cos_lon_ref = np.cos(lon_ref)

#     R_ecef_to_enu = np.array([
#         [-sin_lon_ref,                cos_lon_ref,               0           ],
#         [-sin_lat_ref * cos_lon_ref, -sin_lat_ref * sin_lon_ref, cos_lat_ref ],
#         [ cos_lat_ref * cos_lon_ref,  cos_lat_ref * sin_lon_ref, sin_lat_ref ]
#     ])

#     if ecef_points.ndim == 1:
#         delta_ecef = ecef_points - ref_ecef
#         enu_points = R_ecef_to_enu @ delta_ecef
#     else:
#         delta_ecef = ecef_points - ref_ecef.reshape(1, 3)
#         enu_points = (R_ecef_to_enu @ delta_ecef.T).T
        
#     return enu_points


# def calculate_ecef_errors(gt_poses, pred_poses):
#     """
#     计算基于 ECEF 坐标的定位误差，并收集每个图像的详细数据。
#     """
#     common_images = sorted(list(set(gt_poses.keys()) & set(pred_poses.keys())))
    
#     if not common_images:
#         print("错误: 真值和预测之间没有共同的图像。")
#         return None, None, None, None, None 

#     print(f"找到 {len(common_images)} 个共同图像用于评估。")
#     if len(common_images) < 2 and len(gt_poses) > 0 :
#          print(f"警告: 只有 {len(common_images)} 个共同图像。轨迹图可能不具代表性。")

#     abs_errors_ecef_list = []
#     horizontal_errors_enu_list = []
#     vertical_errors_enu_list = []
    
#     gt_positions_ecef_all = []
#     pred_positions_ecef_all = []
    
#     detailed_results = [] 

#     temp_gt_ecef_for_enu = []
#     temp_pred_ecef_for_enu = []

#     for name in common_images:
#         gt_data_point = gt_poses[name]
#         pred_data_point = pred_poses[name]

#         gt_pos_ecef = gt_data_point['pos_ecef']
#         pred_pos_ecef = pred_data_point['pos_ecef']
        
#         gt_positions_ecef_all.append(gt_pos_ecef)
#         pred_positions_ecef_all.append(pred_pos_ecef)

#         temp_gt_ecef_for_enu.append(gt_pos_ecef)
#         temp_pred_ecef_for_enu.append(pred_pos_ecef)

#         abs_err_ecef = np.linalg.norm(gt_pos_ecef - pred_pos_ecef)
#         abs_errors_ecef_list.append(abs_err_ecef)
        
#         # 从真值数据中获取时间戳和相机ID（如果存在）
#         timestamp_ns = gt_data_point.get('timestamp_ns', 'N/A') # 使用 get 提供默认值
#         camera_id = gt_data_point.get('camera_id', 'N/A')

#         detailed_results.append({
#             'image_name': name,
#             'timestamp_ns': timestamp_ns,
#             'camera_id': camera_id,
#             'gt_qw': gt_data_point['q'][0], 'gt_qx': gt_data_point['q'][1], 
#             'gt_qy': gt_data_point['q'][2], 'gt_qz': gt_data_point['q'][3],
#             'gt_pos_x': gt_pos_ecef[0], 'gt_pos_y': gt_pos_ecef[1], 'gt_pos_z': gt_pos_ecef[2],
#             'pred_qw': pred_data_point['q'][0], 'pred_qx': pred_data_point['q'][1],
#             'pred_qy': pred_data_point['q'][2], 'pred_qz': pred_data_point['q'][3],
#             'pred_pos_x': pred_pos_ecef[0], 'pred_pos_y': pred_pos_ecef[1], 'pred_pos_z': pred_pos_ecef[2],
#             'abs_error_ecef': abs_err_ecef,
#             'horizontal_error_enu': float('nan'), 
#             'vertical_error_enu': float('nan')    
#         })

#     gt_positions_ecef_np = np.array(gt_positions_ecef_all)
#     pred_positions_ecef_np = np.array(pred_positions_ecef_all)
    
#     temp_gt_ecef_np = np.array(temp_gt_ecef_for_enu)
#     temp_pred_ecef_np = np.array(temp_pred_ecef_for_enu)


#     if len(temp_gt_ecef_np) > 0:
#         ref_ecef_for_enu = temp_gt_ecef_np[0]
#         gt_enu_all = ecef_to_enu(temp_gt_ecef_np, ref_ecef_for_enu)
#         pred_enu_all = ecef_to_enu(temp_pred_ecef_np, ref_ecef_for_enu)

#         for i in range(len(gt_enu_all)):
#             horizontal_err = np.linalg.norm(gt_enu_all[i, :2] - pred_enu_all[i, :2])
#             vertical_err = np.abs(gt_enu_all[i, 2] - pred_enu_all[i, 2])
            
#             horizontal_errors_enu_list.append(horizontal_err)
#             vertical_errors_enu_list.append(vertical_err)
            
#             detailed_results[i]['horizontal_error_enu'] = horizontal_err
#             detailed_results[i]['vertical_error_enu'] = vertical_err
#     else:
#         print("警告: 没有共同的点来计算 ENU 误差或绘制轨迹。")

#     abs_errors_ecef_np = np.array(abs_errors_ecef_list)
#     horizontal_errors_enu_np = np.array(horizontal_errors_enu_list)
#     vertical_errors_enu_np = np.array(vertical_errors_enu_list)

#     metrics = {
#         "ATE (Absolute Trajectory Error, ECEF) Mean (m)": np.mean(abs_errors_ecef_np) if len(abs_errors_ecef_np) > 0 else float('nan'),
#         "ATE (ECEF) RMSE (m)": np.sqrt(np.mean(abs_errors_ecef_np**2)) if len(abs_errors_ecef_np) > 0 else float('nan'),
#         "Horizontal Error (ENU) Mean (m)": np.mean(horizontal_errors_enu_np) if len(horizontal_errors_enu_np) > 0 else float('nan'),
#         "Horizontal Error (ENU) RMSE (m)": np.sqrt(np.mean(horizontal_errors_enu_np**2)) if len(horizontal_errors_enu_np) > 0 else float('nan'),
#         "Vertical Error (ENU) Mean (m)": np.mean(vertical_errors_enu_np) if len(vertical_errors_enu_np) > 0 else float('nan'),
#         "Vertical Error (ENU) RMSE (m)": np.sqrt(np.mean(vertical_errors_enu_np**2)) if len(vertical_errors_enu_np) > 0 else float('nan'),
#         "Max Absolute Error (ECEF) (m)": np.max(abs_errors_ecef_np) if len(abs_errors_ecef_np) > 0 else float('nan'),
#         "Max Horizontal Error (ENU) (m)": np.max(horizontal_errors_enu_np) if len(horizontal_errors_enu_np) > 0 else float('nan'),
#         "Max Vertical Error (ENU) (m)": np.max(vertical_errors_enu_np) if len(vertical_errors_enu_np) > 0 else float('nan'),
#         "Number of Common Poses": len(common_images)
#     }
    
#     return metrics, gt_positions_ecef_np, pred_positions_ecef_np, common_images, detailed_results


# def save_detailed_results_to_csv(detailed_results, output_csv_file):
#     """
#     将每个图像的详细结果保存到 CSV 文件。
#     """
#     if not detailed_results:
#         print("没有详细结果可保存。")
#         return

#     # 更新 fieldnames 以包含新的列
#     fieldnames = [
#         'image_name', 'timestamp_ns', 'camera_id',
#         'gt_qw', 'gt_qx', 'gt_qy', 'gt_qz',
#         'gt_pos_x', 'gt_pos_y', 'gt_pos_z',
#         'pred_qw', 'pred_qx', 'pred_qy', 'pred_qz',
#         'pred_pos_x', 'pred_pos_y', 'pred_pos_z',
#         'abs_error_ecef', 'horizontal_error_enu', 'vertical_error_enu'
#     ]

#     try:
#         with open(output_csv_file, 'w', newline='') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             for row_data in detailed_results:
#                 writer.writerow(row_data)
#         print(f"详细结果已保存到: {output_csv_file}")
#     except IOError:
#         print(f"错误: 无法写入 CSV 文件 {output_csv_file}")
#     except Exception as e:
#         print(f"保存 CSV 时发生未知错误: {e}")


# def plot_trajectories_2d_enu(gt_positions_ecef, pred_positions_ecef, common_image_names, title_suffix=""):
#     if gt_positions_ecef.shape[0] == 0 or pred_positions_ecef.shape[0] == 0:
#         print("没有足够的轨迹点进行 2D ENU 绘图。")
#         return

#     ref_ecef_for_enu = gt_positions_ecef[0]
#     gt_enu = ecef_to_enu(gt_positions_ecef, ref_ecef_for_enu)
#     pred_enu = ecef_to_enu(pred_positions_ecef, ref_ecef_for_enu)

#     plt.figure(figsize=(10, 8))
#     if gt_enu.shape[0] > 1:
#         plt.plot(gt_enu[:, 0], gt_enu[:, 1], 'g-', label='真值轨迹 (Ground Truth)', linewidth=2)
#         plt.plot(gt_enu[0, 0], gt_enu[0, 1], 'go', markersize=8, label='GT Start')
#         plt.plot(gt_enu[-1, 0], gt_enu[-1, 1], 'gx', markersize=8, label='GT End')
#     elif gt_enu.shape[0] == 1: # 如果只有一个点
#         plt.plot(gt_enu[0, 0], gt_enu[0, 1], 'go', markersize=8, label='GT Pose')


#     if pred_enu.shape[0] > 1:
#         plt.plot(pred_enu[:, 0], pred_enu[:, 1], 'r--', label='预测轨迹 (Predicted)', linewidth=2)
#         plt.plot(pred_enu[0, 0], pred_enu[0, 1], 'ro', markersize=8, label='Pred Start')
#         plt.plot(pred_enu[-1, 0], pred_enu[-1, 1], 'rx', markersize=8, label='Pred End')
#     elif pred_enu.shape[0] == 1:
#         plt.plot(pred_enu[0, 0], pred_enu[0, 1], 'ro', markersize=8, label='Pred Pose')


#     plt.xlabel('East (m)')
#     plt.ylabel('North (m)')
#     plt.title(f'轨迹比较 (ENU 俯视图) {title_suffix}')
#     plt.legend()
#     plt.grid(True)
#     plt.axis('equal')
#     plt.show()

# def plot_trajectories_3d_ecef(gt_positions_ecef, pred_positions_ecef, title_suffix=""):
#     if gt_positions_ecef.shape[0] == 0 or pred_positions_ecef.shape[0] == 0:
#         print("没有足够的轨迹点进行 3D ECEF 绘图。")
#         return

#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     if gt_positions_ecef.shape[0] > 1:
#         ax.plot(gt_positions_ecef[:, 0], gt_positions_ecef[:, 1], gt_positions_ecef[:, 2], 
#                 'g-', label='真值轨迹 (Ground Truth)', linewidth=2)
#         ax.scatter(gt_positions_ecef[0, 0], gt_positions_ecef[0, 1], gt_positions_ecef[0, 2], 
#                    c='green', marker='o', s=50, label='GT Start')
#         ax.scatter(gt_positions_ecef[-1, 0], gt_positions_ecef[-1, 1], gt_positions_ecef[-1, 2], 
#                    c='green', marker='x', s=50, label='GT End')
#     elif gt_positions_ecef.shape[0] == 1:
#          ax.scatter(gt_positions_ecef[0, 0], gt_positions_ecef[0, 1], gt_positions_ecef[0, 2], 
#                    c='green', marker='o', s=50, label='GT Pose')


#     if pred_positions_ecef.shape[0] > 1:
#         ax.plot(pred_positions_ecef[:, 0], pred_positions_ecef[:, 1], pred_positions_ecef[:, 2], 
#                 'r--', label='预测轨迹 (Predicted)', linewidth=2)
#         ax.scatter(pred_positions_ecef[0, 0], pred_positions_ecef[0, 1], pred_positions_ecef[0, 2], 
#                    c='red', marker='o', s=50, label='Pred Start')
#         ax.scatter(pred_positions_ecef[-1, 0], pred_positions_ecef[-1, 1], pred_positions_ecef[-1, 2], 
#                    c='red', marker='x', s=50, label='Pred End')
#     elif pred_positions_ecef.shape[0] == 1:
#         ax.scatter(pred_positions_ecef[0, 0], pred_positions_ecef[0, 1], pred_positions_ecef[0, 2], 
#                    c='red', marker='o', s=50, label='Pred Pose')


#     ax.set_xlabel('ECEF X (m)')
#     ax.set_ylabel('ECEF Y (m)')
#     ax.set_zlabel('ECEF Z (m)')
#     ax.set_title(f'3D 轨迹比较 (ECEF) {title_suffix}')
#     ax.legend()
    
#     # 尝试使各轴比例大致相等
#     all_coords = np.vstack((gt_positions_ecef, pred_positions_ecef))
#     if all_coords.size > 0 : # 确保有数据点
#         min_coords = np.min(all_coords, axis=0)
#         max_coords = np.max(all_coords, axis=0)
#         center = (min_coords + max_coords) / 2
#         plot_radius = 0.5 * np.max(max_coords - min_coords)
#         if plot_radius == 0: plot_radius = 1.0 # 避免半径为0的情况
#         ax.set_xlim([center[0] - plot_radius, center[0] + plot_radius])
#         ax.set_ylim([center[1] - plot_radius, center[1] + plot_radius])
#         ax.set_zlim([center[2] - plot_radius, center[2] + plot_radius])
    
#     plt.show()


# if __name__ == "__main__":
#     # --- 文件路径配置 ---
#     # !! 请确保将此路径替换为您的实际真值 CSV 文件路径 !!
#     ground_truth_file = "datasets/SensorKit/query_camera_ecef_poses_with_quaternion.csv"
#     # 确保您的预测文件名与 SensorKit.py 中的输出一致
#     predicted_file = "outputs/SensorKit_hloc/only_scale/SensorKit_query_poses_fully_transformed.txt" 
#     # 定义输出详细结果的 CSV 文件名
#     output_detailed_csv = "outputs/SensorKit_hloc/only_scale/detailed_evaluation_results.csv" 

#     print(f"正在读取真值 CSV 文件: {ground_truth_file}")
#     gt_data = read_gt_csv_pose_file(ground_truth_file)
    
#     print(f"正在读取预测 TXT 文件: {predicted_file}")
#     pred_data = read_pred_txt_pose_file(predicted_file)

#     if not gt_data or not pred_data:
#         print("错误: 无法加载一个或两个位姿文件。正在退出。")
#         exit()

#     print("\n--- 计算误差 ---")
#     # 确保接收 calculate_ecef_errors 返回的所有五个值
#     error_metrics, gt_traj_ecef, pred_traj_ecef, common_names, detailed_results_list = calculate_ecef_errors(gt_data, pred_data)

#     if error_metrics:
#         print("\n--- 定位误差指标 ---")
#         for metric, value in error_metrics.items():
#             if isinstance(value, float):
#                 print(f"{metric}: {value:.3f}")
#             else:
#                 print(f"{metric}: {value}")
        
#         # --- 先写入 CSV 文件 ---
#         if detailed_results_list:
#             # 确保输出目录存在
#             output_dir = os.path.dirname(output_detailed_csv)
#             if output_dir and not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#                 print(f"创建目录: {output_dir}")
#             save_detailed_results_to_csv(detailed_results_list, output_detailed_csv)
#         else:
#             print("没有详细结果可保存到 CSV。")
            
#     else:
#         print("未能计算误差指标。")

#     # --- 然后进行可视化 ---
#     if gt_traj_ecef is not None and pred_traj_ecef is not None and len(gt_traj_ecef) > 0:
#         print("\n--- 可视化轨迹 ---")
#         plot_trajectories_2d_enu(gt_traj_ecef, pred_traj_ecef, common_names, title_suffix="Localization")
#         plot_trajectories_3d_ecef(gt_traj_ecef, pred_traj_ecef, title_suffix="Localization")
#     else:
#         print("没有足够的匹配轨迹数据进行可视化。")







import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import csv 
import os

# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'FangSong'] # 优先使用文泉驿正黑, 然后是黑体, 然后是仿宋
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

def read_gt_csv_pose_file(filepath):
    """
    读取包含 ECEF 位姿的 CSV 真值文件。
    CSV 列应包含: image_path, quat_w_c2e, quat_x_c2e, quat_y_c2e, quat_z_c2e, 
                     pos_x_ecef, pos_y_ecef, pos_z_ecef,
                     image_timestamp_ns, camera_id (可选)
    """
    poses = {}
    expected_headers_minimal = ['image_path', 'quat_w_c2e', 'quat_x_c2e', 'quat_y_c2e', 'quat_z_c2e',
                                'pos_x_ecef', 'pos_y_ecef', 'pos_z_ecef']
    
    try:
        with open(filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            actual_fieldnames = reader.fieldnames
            if actual_fieldnames is None:
                print(f"错误: CSV 文件 {filepath} 为空或无法读取表头。")
                return poses

            if not all(header in actual_fieldnames for header in expected_headers_minimal):
                print(f"错误: CSV 文件 {filepath} 的表头缺少必要的列。")
                print(f"预期至少包含: {expected_headers_minimal}")
                print(f"实际表头: {actual_fieldnames}")
                return poses

            has_timestamp = 'image_timestamp_ns' in actual_fieldnames
            has_camera_id = 'camera_id' in actual_fieldnames

            for row in reader:
                try:
                    image_full_path = row['image_path']
                    name = os.path.basename(image_full_path)
                    
                    q_val = np.array([
                        float(row['quat_w_c2e']), 
                        float(row['quat_x_c2e']), 
                        float(row['quat_y_c2e']), 
                        float(row['quat_z_c2e'])
                    ])
                    pos_ecef_val = np.array([
                        float(row['pos_x_ecef']), 
                        float(row['pos_y_ecef']), 
                        float(row['pos_z_ecef'])
                    ])
                    
                    pose_data = {'q': q_val, 'pos_ecef': pos_ecef_val}
                    
                    if has_timestamp:
                        pose_data['timestamp_ns'] = row.get('image_timestamp_ns', '')
                    if has_camera_id:
                        pose_data['camera_id'] = row.get('camera_id', '')

                    poses[name] = pose_data
                except ValueError as e:
                    print(f"警告: 解析 CSV 行时出错: {row} - {e}")
                    continue
                except KeyError as e:
                    print(f"警告: CSV 行中缺少预期的键: {e} 在行 {row}")
                    continue
    except FileNotFoundError:
        print(f"错误: 找不到真值文件 {filepath}")
    except Exception as e:
        print(f"读取或解析 CSV 文件 {filepath} 时发生错误: {e}")
    return poses

def read_pred_txt_pose_file(filepath):
    """
    读取预测的 TXT 位姿文件。
    文件格式: image_name qw qx qy qz pos_x pos_y pos_z (pos_x, pos_y, pos_z 是 ECEF)
    """
    poses = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) != 8:
                    print(f"警告 (预测文件): 跳过格式不正确的行: {line.strip()}")
                    continue
                
                name = parts[0]
                try:
                    q = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                    pos_ecef = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
                    poses[name] = {'q': q, 'pos_ecef': pos_ecef}
                except ValueError as e:
                    print(f"警告 (预测文件): 解析行时出错: {line.strip()} - {e}")
                    continue
    except FileNotFoundError:
        print(f"错误: 找不到预测文件 {filepath}")
    except Exception as e:
        print(f"读取或解析预测文件 {filepath} 时发生错误: {e}")
    return poses


def ecef_to_enu(ecef_points, ref_ecef):
    """
    将 ECEF 坐标点（或点列表）转换为相对于参考 ECEF 点的本地 ENU 坐标。
    """
    a = 6378137.0 
    e_sq = 6.69437999014e-3

    ref_x, ref_y, ref_z = ref_ecef
    p_ref = np.sqrt(ref_x**2 + ref_y**2)
    
    lon_ref = np.arctan2(ref_y, ref_x)
    lat_ref = np.arctan2(ref_z, p_ref * (1 - e_sq)) 
    for _ in range(10): 
        sin_lat_ref = np.sin(lat_ref)
        N_ref = a / np.sqrt(1 - e_sq * sin_lat_ref**2)
        h_ref = p_ref / np.cos(lat_ref) - N_ref 
        lat_ref_new = np.arctan2(ref_z, p_ref * (1 - e_sq * N_ref / (N_ref + h_ref)))
        if np.abs(lat_ref_new - lat_ref) < 1e-9: 
            lat_ref = lat_ref_new
            break
        lat_ref = lat_ref_new

    sin_lat_ref = np.sin(lat_ref)
    cos_lat_ref = np.cos(lat_ref)
    sin_lon_ref = np.sin(lon_ref)
    cos_lon_ref = np.cos(lon_ref)

    R_ecef_to_enu = np.array([
        [-sin_lon_ref,                cos_lon_ref,               0           ],
        [-sin_lat_ref * cos_lon_ref, -sin_lat_ref * sin_lon_ref, cos_lat_ref ],
        [ cos_lat_ref * cos_lon_ref,  cos_lat_ref * sin_lon_ref, sin_lat_ref ]
    ])

    if ecef_points.ndim == 1:
        delta_ecef = ecef_points - ref_ecef
        enu_points = R_ecef_to_enu @ delta_ecef
    else:
        delta_ecef = ecef_points - ref_ecef.reshape(1, 3)
        enu_points = (R_ecef_to_enu @ delta_ecef.T).T
        
    return enu_points


def calculate_ecef_errors(gt_poses, pred_poses):
    """
    计算基于 ECEF 坐标的定位误差，并收集每个图像的详细数据。
    """
    common_images = sorted(list(set(gt_poses.keys()) & set(pred_poses.keys())))
    
    if not common_images:
        print("错误: 真值和预测之间没有共同的图像。")
        return None, None, None, None, None 

    print(f"找到 {len(common_images)} 个共同图像用于评估。")
    if len(common_images) < 2 and len(gt_poses) > 0 :
         print(f"警告: 只有 {len(common_images)} 个共同图像。轨迹图可能不具代表性。")

    abs_errors_ecef_list = []
    horizontal_errors_enu_list = []
    vertical_errors_enu_list = []
    
    gt_positions_ecef_all = []
    pred_positions_ecef_all = []
    
    detailed_results = [] 

    temp_gt_ecef_for_enu = []
    temp_pred_ecef_for_enu = []

    for name in common_images:
        gt_data_point = gt_poses[name]
        pred_data_point = pred_poses[name]

        gt_pos_ecef = gt_data_point['pos_ecef']
        pred_pos_ecef = pred_data_point['pos_ecef']
        
        gt_positions_ecef_all.append(gt_pos_ecef)
        pred_positions_ecef_all.append(pred_pos_ecef)

        temp_gt_ecef_for_enu.append(gt_pos_ecef)
        temp_pred_ecef_for_enu.append(pred_pos_ecef)

        abs_err_ecef = np.linalg.norm(gt_pos_ecef - pred_pos_ecef)
        abs_errors_ecef_list.append(abs_err_ecef)
        
        timestamp_ns = gt_data_point.get('timestamp_ns', 'N/A') 
        camera_id = gt_data_point.get('camera_id', 'N/A')

        detailed_results.append({
            'image_name': name,
            'timestamp_ns': timestamp_ns,
            'camera_id': camera_id,
            'gt_qw': gt_data_point['q'][0], 'gt_qx': gt_data_point['q'][1], 
            'gt_qy': gt_data_point['q'][2], 'gt_qz': gt_data_point['q'][3],
            'gt_pos_x': gt_pos_ecef[0], 'gt_pos_y': gt_pos_ecef[1], 'gt_pos_z': gt_pos_ecef[2],
            'pred_qw': pred_data_point['q'][0], 'pred_qx': pred_data_point['q'][1],
            'pred_qy': pred_data_point['q'][2], 'pred_qz': pred_data_point['q'][3],
            'pred_pos_x': pred_pos_ecef[0], 'pred_pos_y': pred_pos_ecef[1], 'pred_pos_z': pred_pos_ecef[2],
            'abs_error_ecef': abs_err_ecef,
            'horizontal_error_enu': float('nan'), 
            'vertical_error_enu': float('nan')    
        })

    gt_positions_ecef_np = np.array(gt_positions_ecef_all)
    pred_positions_ecef_np = np.array(pred_positions_ecef_all)
    
    temp_gt_ecef_np = np.array(temp_gt_ecef_for_enu)
    temp_pred_ecef_np = np.array(temp_pred_ecef_for_enu)

    if len(temp_gt_ecef_np) > 0:
        ref_ecef_for_enu = temp_gt_ecef_np[0]
        gt_enu_all = ecef_to_enu(temp_gt_ecef_np, ref_ecef_for_enu)
        pred_enu_all = ecef_to_enu(temp_pred_ecef_np, ref_ecef_for_enu)

        for i in range(len(gt_enu_all)):
            horizontal_err = np.linalg.norm(gt_enu_all[i, :2] - pred_enu_all[i, :2])
            vertical_err = np.abs(gt_enu_all[i, 2] - pred_enu_all[i, 2])
            
            horizontal_errors_enu_list.append(horizontal_err)
            vertical_errors_enu_list.append(vertical_err)
            
            detailed_results[i]['horizontal_error_enu'] = horizontal_err
            detailed_results[i]['vertical_error_enu'] = vertical_err
    else:
        print("警告: 没有共同的点来计算 ENU 误差或绘制轨迹。")

    abs_errors_ecef_np = np.array(abs_errors_ecef_list)
    horizontal_errors_enu_np = np.array(horizontal_errors_enu_list)
    vertical_errors_enu_np = np.array(vertical_errors_enu_list)

    metrics = {
        "ATE (Absolute Trajectory Error, ECEF) Mean (m)": np.mean(abs_errors_ecef_np) if len(abs_errors_ecef_np) > 0 else float('nan'),
        "ATE (ECEF) RMSE (m)": np.sqrt(np.mean(abs_errors_ecef_np**2)) if len(abs_errors_ecef_np) > 0 else float('nan'),
        "Horizontal Error (ENU) Mean (m)": np.mean(horizontal_errors_enu_np) if len(horizontal_errors_enu_np) > 0 else float('nan'),
        "Horizontal Error (ENU) RMSE (m)": np.sqrt(np.mean(horizontal_errors_enu_np**2)) if len(horizontal_errors_enu_np) > 0 else float('nan'),
        "Vertical Error (ENU) Mean (m)": np.mean(vertical_errors_enu_np) if len(vertical_errors_enu_np) > 0 else float('nan'),
        "Vertical Error (ENU) RMSE (m)": np.sqrt(np.mean(vertical_errors_enu_np**2)) if len(vertical_errors_enu_np) > 0 else float('nan'),
        "Max Absolute Error (ECEF) (m)": np.max(abs_errors_ecef_np) if len(abs_errors_ecef_np) > 0 else float('nan'),
        "Max Horizontal Error (ENU) (m)": np.max(horizontal_errors_enu_np) if len(horizontal_errors_enu_np) > 0 else float('nan'),
        "Max Vertical Error (ENU) (m)": np.max(vertical_errors_enu_np) if len(vertical_errors_enu_np) > 0 else float('nan'),
        "Number of Common Poses": len(common_images)
    }
    
    return metrics, gt_positions_ecef_np, pred_positions_ecef_np, common_images, detailed_results


def save_detailed_results_to_csv(detailed_results, output_csv_file):
    """
    将每个图像的详细结果保存到 CSV 文件。
    """
    if not detailed_results:
        print("没有详细结果可保存。")
        return

    fieldnames = [
        'image_name', 'timestamp_ns', 'camera_id',
        'gt_qw', 'gt_qx', 'gt_qy', 'gt_qz',
        'gt_pos_x', 'gt_pos_y', 'gt_pos_z',
        'pred_qw', 'pred_qx', 'pred_qy', 'pred_qz',
        'pred_pos_x', 'pred_pos_y', 'pred_pos_z',
        'abs_error_ecef', 'horizontal_error_enu', 'vertical_error_enu'
    ]

    try:
        with open(output_csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row_data in detailed_results:
                writer.writerow(row_data)
        print(f"详细结果已保存到: {output_csv_file}")
    except IOError as e: # 更具体地捕获IOError
        print(f"错误: 无法写入 CSV 文件 {output_csv_file} - {e}")
    except Exception as e:
        print(f"保存 CSV 时发生未知错误: {e}")


def plot_trajectories_2d_enu(gt_positions_ecef, pred_positions_ecef, common_image_names, detailed_results, title_suffix=""):
    if gt_positions_ecef.shape[0] == 0 or pred_positions_ecef.shape[0] == 0:
        print("没有足够的轨迹点进行 2D ENU 绘图。")
        return

    points_to_plot_gt = []
    points_to_plot_pred = []
    error_threshold = 10.0 

    # 确保 detailed_results 与 common_image_names (以及轨迹点) 对齐
    # common_image_names 应该与 gt_positions_ecef 和 pred_positions_ecef 的行数相同
    if len(detailed_results) != gt_positions_ecef.shape[0] or len(detailed_results) == 0:
        print("警告 (2D Plot): detailed_results 与轨迹点数不匹配或为空，将绘制所有点。")
        points_to_plot_gt = gt_positions_ecef
        points_to_plot_pred = pred_positions_ecef
    else:
        for i, result in enumerate(detailed_results):
            # 确保 result 字典中有 abs_error_ecef 键
            if result.get('abs_error_ecef', float('inf')) <= error_threshold:
                points_to_plot_gt.append(gt_positions_ecef[i])
                points_to_plot_pred.append(pred_positions_ecef[i])
        
        points_to_plot_gt = np.array(points_to_plot_gt)
        points_to_plot_pred = np.array(points_to_plot_pred)

    if points_to_plot_gt.shape[0] == 0:
        print("2D ENU Plot: 根据误差阈值过滤后没有点可绘制。")
        return
    
    ref_ecef_for_enu = points_to_plot_gt[0] 
    gt_enu = ecef_to_enu(points_to_plot_gt, ref_ecef_for_enu)
    pred_enu = ecef_to_enu(points_to_plot_pred, ref_ecef_for_enu)

    plt.figure(figsize=(10, 8))
    plot_label_gt = '真值轨迹 (误差 <= 10m)' if len(detailed_results) > 0 and len(detailed_results) == gt_positions_ecef.shape[0] else '真值轨迹'
    plot_label_pred = '预测轨迹 (误差 <= 10m)' if len(detailed_results) > 0 and len(detailed_results) == gt_positions_ecef.shape[0] else '预测轨迹'

    if gt_enu.shape[0] > 1:
        plt.plot(gt_enu[:, 0], gt_enu[:, 1], 'g-', label=plot_label_gt, linewidth=2)
        plt.plot(gt_enu[0, 0], gt_enu[0, 1], 'go', markersize=8, label='GT Start (Filtered)')
        plt.plot(gt_enu[-1, 0], gt_enu[-1, 1], 'gx', markersize=8, label='GT End (Filtered)')
    elif gt_enu.shape[0] == 1: 
        plt.plot(gt_enu[0, 0], gt_enu[0, 1], 'go', markersize=8, label='GT Pose (Filtered)')

    if pred_enu.shape[0] > 1:
        plt.plot(pred_enu[:, 0], pred_enu[:, 1], 'r--', label=plot_label_pred, linewidth=2)
        plt.plot(pred_enu[0, 0], pred_enu[0, 1], 'ro', markersize=8, label='Pred Start (Filtered)')
        plt.plot(pred_enu[-1, 0], pred_enu[-1, 1], 'rx', markersize=8, label='Pred End (Filtered)')
    elif pred_enu.shape[0] == 1:
        plt.plot(pred_enu[0, 0], pred_enu[0, 1], 'ro', markersize=8, label='Pred Pose (Filtered)')

    plt.xlabel('East (m)')
    plt.ylabel('North (m)')
    plt.title(f'轨迹比较 (ENU 俯视图) {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_trajectories_3d_ecef(gt_positions_ecef, pred_positions_ecef, detailed_results, title_suffix=""):
    if gt_positions_ecef.shape[0] == 0 or pred_positions_ecef.shape[0] == 0:
        print("没有足够的轨迹点进行 3D ECEF 绘图。")
        return

    points_to_plot_gt_3d = []
    points_to_plot_pred_3d = []
    error_threshold = 10.0

    if len(detailed_results) != gt_positions_ecef.shape[0] or len(detailed_results) == 0:
        print("警告 (3D Plot): detailed_results 与轨迹点数不匹配或为空，将绘制所有点。")
        points_to_plot_gt_3d = gt_positions_ecef
        points_to_plot_pred_3d = pred_positions_ecef
    else:
        for i, result in enumerate(detailed_results):
            if result.get('abs_error_ecef', float('inf')) <= error_threshold:
                points_to_plot_gt_3d.append(gt_positions_ecef[i])
                points_to_plot_pred_3d.append(pred_positions_ecef[i])
        
        points_to_plot_gt_3d = np.array(points_to_plot_gt_3d)
        points_to_plot_pred_3d = np.array(points_to_plot_pred_3d)

    if points_to_plot_gt_3d.shape[0] == 0:
        print("3D ECEF Plot: 根据误差阈值过滤后没有点可绘制。")
        return
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    plot_label_gt_3d = '真值轨迹 (误差 <= 10m)' if len(detailed_results) > 0 and len(detailed_results) == gt_positions_ecef.shape[0] else '真值轨迹'
    plot_label_pred_3d = '预测轨迹 (误差 <= 10m)' if len(detailed_results) > 0 and len(detailed_results) == gt_positions_ecef.shape[0] else '预测轨迹'

    if points_to_plot_gt_3d.shape[0] > 1:
        ax.plot(points_to_plot_gt_3d[:, 0], points_to_plot_gt_3d[:, 1], points_to_plot_gt_3d[:, 2], 
                'g-', label=plot_label_gt_3d, linewidth=2)
        ax.scatter(points_to_plot_gt_3d[0, 0], points_to_plot_gt_3d[0, 1], points_to_plot_gt_3d[0, 2], 
                   c='green', marker='o', s=50, label='GT Start (Filtered)')
        ax.scatter(points_to_plot_gt_3d[-1, 0], points_to_plot_gt_3d[-1, 1], points_to_plot_gt_3d[-1, 2], 
                   c='green', marker='x', s=50, label='GT End (Filtered)')
    elif points_to_plot_gt_3d.shape[0] == 1:
         ax.scatter(points_to_plot_gt_3d[0, 0], points_to_plot_gt_3d[0, 1], points_to_plot_gt_3d[0, 2], 
                   c='green', marker='o', s=50, label='GT Pose (Filtered)')

    if points_to_plot_pred_3d.shape[0] > 1:
        ax.plot(points_to_plot_pred_3d[:, 0], points_to_plot_pred_3d[:, 1], points_to_plot_pred_3d[:, 2], 
                'r--', label=plot_label_pred_3d, linewidth=2)
        ax.scatter(points_to_plot_pred_3d[0, 0], points_to_plot_pred_3d[0, 1], points_to_plot_pred_3d[0, 2], 
                   c='red', marker='o', s=50, label='Pred Start (Filtered)')
        ax.scatter(points_to_plot_pred_3d[-1, 0], points_to_plot_pred_3d[-1, 1], points_to_plot_pred_3d[-1, 2], 
                   c='red', marker='x', s=50, label='Pred End (Filtered)')
    elif points_to_plot_pred_3d.shape[0] == 1:
        ax.scatter(points_to_plot_pred_3d[0, 0], points_to_plot_pred_3d[0, 1], points_to_plot_pred_3d[0, 2], 
                   c='red', marker='o', s=50, label='Pred Pose (Filtered)')

    ax.set_xlabel('ECEF X (m)')
    ax.set_ylabel('ECEF Y (m)')
    ax.set_zlabel('ECEF Z (m)')
    ax.set_title(f'3D 轨迹比较 (ECEF) {title_suffix}')
    ax.legend()
    
    all_coords_filtered = np.vstack((points_to_plot_gt_3d, points_to_plot_pred_3d)) 
    if all_coords_filtered.size > 0 : 
        min_coords = np.min(all_coords_filtered, axis=0)
        max_coords = np.max(all_coords_filtered, axis=0)
        center = (min_coords + max_coords) / 2
        plot_radius = 0.5 * np.max(max_coords - min_coords)
        if plot_radius == 0: plot_radius = 1.0 
        ax.set_xlim([center[0] - plot_radius, center[0] + plot_radius])
        ax.set_ylim([center[1] - plot_radius, center[1] + plot_radius])
        ax.set_zlim([center[2] - plot_radius, center[2] + plot_radius])
    
    plt.show()


if __name__ == "__main__":
    # --- 文件路径配置 ---
    ground_truth_file = "datasets/SensorKit/query_camera_ecef_poses_with_quaternion.csv"
    predicted_file = "outputs/SensorKit_hloc/only_scale/SensorKit_query_poses_fully_transformed.txt" 
    output_detailed_csv = "outputs/SensorKit_hloc/only_scale/detailed_evaluation_results.csv" 

    print(f"正在读取真值 CSV 文件: {ground_truth_file}")
    gt_data = read_gt_csv_pose_file(ground_truth_file)
    
    print(f"正在读取预测 TXT 文件: {predicted_file}")
    pred_data = read_pred_txt_pose_file(predicted_file)

    if not gt_data or not pred_data:
        print("错误: 无法加载一个或两个位姿文件。正在退出。")
        exit()

    print("\n--- 计算误差 ---")
    error_metrics, gt_traj_ecef, pred_traj_ecef, common_names, detailed_results_list = calculate_ecef_errors(gt_data, pred_data)

    if error_metrics:
        print("\n--- 定位误差指标 ---")
        for metric, value in error_metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.3f}")
            else:
                print(f"{metric}: {value}")
        
        if detailed_results_list:
            output_dir = os.path.dirname(output_detailed_csv)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                    print(f"创建目录: {output_dir}")
                except OSError as e:
                    print(f"错误: 创建目录 {output_dir} 失败 - {e}")
                    # 决定是否在此处退出或继续而不保存CSV
            
            # 仅当目录存在或成功创建时才尝试保存
            if not output_dir or os.path.exists(output_dir):
                 save_detailed_results_to_csv(detailed_results_list, output_detailed_csv)
            else:
                print(f"由于目录创建失败，跳过保存 CSV 文件到 {output_detailed_csv}")

        else:
            print("没有详细结果可保存到 CSV。")
            
    else:
        print("未能计算误差指标。")

    # --- 然后进行可视化 (只绘制误差 <= 10m 的点) ---
    # 确保 detailed_results_list 存在且非空才进行基于误差的过滤绘图
    if gt_traj_ecef is not None and pred_traj_ecef is not None and len(gt_traj_ecef) > 0 and detailed_results_list and len(detailed_results_list) > 0:
        print("\n--- 可视化轨迹 (误差 <= 10m 的点) ---")
        plot_trajectories_2d_enu(gt_traj_ecef, pred_traj_ecef, common_names, detailed_results_list, title_suffix="Localization (Error <= 10m)")
        plot_trajectories_3d_ecef(gt_traj_ecef, pred_traj_ecef, detailed_results_list, title_suffix="Localization (Error <= 10m)")
    elif gt_traj_ecef is not None and pred_traj_ecef is not None and len(gt_traj_ecef) > 0:
        # 如果 detailed_results_list 不可用或为空，但仍有轨迹数据，则绘制所有点
        print("\n--- 可视化轨迹 (无法按误差过滤或无详细结果，将绘制所有点) ---")
        plot_trajectories_2d_enu(gt_traj_ecef, pred_traj_ecef, common_names, [], title_suffix="Localization (All Points)") # 传递空列表
        plot_trajectories_3d_ecef(gt_traj_ecef, pred_traj_ecef, [], title_suffix="Localization (All Points)") # 传递空列表
    else:
        print("没有足够的匹配轨迹数据进行可视化。")