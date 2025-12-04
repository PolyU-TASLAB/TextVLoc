
import rospy
import rosbag
import cv2
import pickle
from cv_bridge import CvBridge
import numpy as np
import argparse
from pathlib import Path
import h5py
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from io import BytesIO
from read_write_model import read_model, qvec2rotmat
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, TransformStamped
import std_msgs.msg
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Path as RosPath




def save_visualization_as_image(query_image, ref_image, kp_query, kp_ref, inliers, query_name, ref_name):
    try:
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        ax[0].imshow(query_image[..., ::-1])  # BGR 转 RGB
        ax[0].set_title(f"Query: {query_name}")
        ax[0].axis("off")

        ax[1].imshow(ref_image[..., ::-1])  # BGR 转 RGB
        ax[1].set_title(f"Reference: {ref_name}")
        ax[1].axis("off")

        # 将关键点转换为 NumPy 数组（确保输入是二维数组）
        kp_query = np.array(kp_query)
        kp_ref = np.array(kp_ref)

        # 绘制匹配点连线
        for i, (q_pt, r_pt, inlier) in enumerate(zip(kp_query, kp_ref, inliers)):
            color = "g" if inlier else "r"  # 内点为绿色，外点为红色
            line = ConnectionPatch(
                xyA=(q_pt[0], q_pt[1]), coordsA=ax[0].transData,
                xyB=(r_pt[0], r_pt[1]), coordsB=ax[1].transData,
                color=color, linewidth=1.5, alpha=0.8
            )
            fig.add_artist(line)

        # 绘制关键点
        ax[0].scatter(kp_query[:, 0], kp_query[:, 1], c="blue", s=10, label="Query Keypoints")
        ax[1].scatter(kp_ref[:, 0], kp_ref[:, 1], c="orange", s=10, label="Reference Keypoints")

        # 保存图像到缓冲区
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        buf.close()
        plt.close(fig)

        print(f"成功生成可视化图像，查询图像: {query_name}, 参考图像: {ref_name}")
        return img
    except Exception as e:
        print(f"警告: 生成可视化图像时发生错误: {e}")
        return None

def parse_results(path):
    poses = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                # 正确的解析方式
                ts = float(parts[0])  # 时间戳是第一个元素
                tvec = np.array(parts[1:4], float)
                qvec = np.array(parts[4:8], float)[[1, 2, 3, 0]] # xyzw -> wxyz
                # 使用时间戳作为 key
                poses[ts] = (qvec, tvec)
            elif len(parts) == 9:
                # 如果有其他格式，也需要相应修改
                # 这里假设旧逻辑仍然可能需要，但需要确认
                name, ts = Path(parts[0]).stem, 0.0 # 如果没有时间戳，就用0
                qvec, tvec = np.array(parts[1:5], float), np.array(parts[5:8], float)
                # 这种情况下，我们没有一个好的 key，可以暂时跳过
                continue
            else:
                continue
    return poses


def to_pointcloud2_msg(points, stamp, frame_id="map"):
    """将 Nx3 的 numpy 点云数组转换为 PointCloud2 消息"""
    header = std_msgs.msg.Header(stamp=stamp, frame_id=frame_id)
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    # 如果点云是 Nx6 (xyzrgb)，可以添加颜色字段
    if points.shape[1] == 6:
        fields.append(PointField('rgb', 12, PointField.UINT32, 1))
        # 将 RGB (0-1) 转换为单个 uint32
        r = (points[:, 3] * 255).astype(np.uint8)
        g = (points[:, 4] * 255).astype(np.uint8)
        b = (points[:, 5] * 255).astype(np.uint8)
        rgb_packed = np.array((r << 16) | (g << 8) | b, dtype=np.uint32)
        # 重塑数据以匹配 PointCloud2 格式
        new_points = np.zeros(len(points), dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.uint32)])
        new_points['x'], new_points['y'], new_points['z'] = points[:,0], points[:,1], points[:,2]
        new_points['rgb'] = rgb_packed
        return PointCloud2(header=header, height=1, width=len(points), is_dense=True, point_step=16, row_step=16*len(points), fields=fields, data=new_points.tobytes())

    # 仅 XYZ
    return PointCloud2(header=header, height=1, width=len(points), is_dense=True, point_step=12, row_step=12*len(points), fields=fields, data=points.astype(np.float32).tobytes())


def main(args):
    print("正在加载所有数据...")
    cameras, images, points3D = read_model(path=args.sfm_dir, ext=".bin")
    poses_w2c_by_ts = parse_results(args.results_file)
    match_file = h5py.File(args.match_file, 'r')

    # 加载 logs 文件
    logs_path = Path(args.logs_file)  # 从命令行参数获取 logs 文件路径
    if not logs_path.is_file():
        print(f"错误: logs 文件不存在: {logs_path}")
        exit(1)

    with open(logs_path, "rb") as f:
        logs = pickle.load(f)
    print(f"成功加载 logs 文件: {logs_path}")

    # --- 加载并按时间戳排序真值数据 ---
    ground_truth_poses_map = {}
    if args.ground_truth_file and args.ground_truth_file.exists():
        print(f"正在加载真值文件: {args.ground_truth_file}")
        df_gt = pd.read_csv(
            args.ground_truth_file, sep=' ', comment='#', header=None,
            names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
        ).sort_values(by='timestamp')
        for _, row in df_gt.iterrows():
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.from_sec(row['timestamp'])
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = row['tx'], row['ty'], row['tz']
            pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = row['qx'], row['qy'], row['qz'], row['qw']
            ground_truth_poses_map[row['timestamp']] = pose_msg

    # --- 核心修改：重构 query_data 的加载逻辑以适应分层 HDF5 ---
    print("正在从分层 HDF5 特征文件中加载查询数据...")
    query_data = {}

# 打开查询图像特征文件
    query_feature_file = h5py.File(args.query_feature_file, 'r')  # 查询图像特征文件
    ref_feature_file = h5py.File(args.ref_feature_file, 'r')  # 参考图像特征文件
    
    def find_image_groups(name, obj):
        # visititems 的回调函数
        # 我们只关心名字像图像文件的组 (Group)
        if isinstance(obj, h5py.Group) and (name.endswith('.jpg') or name.endswith('.png')):
            try:
                p = Path(name)
                timestamp_str = p.stem
                timestamp_sec = float(timestamp_str) / 1e9
                # key 是完整的 HDF5 路径，value 是 (时间戳, 文件名)
                query_data[name] = (timestamp_sec, p.name)
            except ValueError:
                print(f"警告: 无法从 HDF5 key '{name}' 中解析时间戳，已跳过。")

    query_feature_file.visititems(find_image_groups)
    # 打印查询图像特征文件的结构
    print("查询图像特征文件结构:")
    query_feature_file.visititems(lambda name, obj: print(name))

    if not query_data:
        print("\n错误: 未能从特征文件中加载任何有效的查询图像数据。")
        print("请检查您的 query_feature_file HDF5 结构是否正确，它应该包含类似 '12345.jpg' 的路径。")
        exit(1)

    print(f"成功加载 {len(query_data)} 个查询帧。")

    bridge = CvBridge()

    # 2. 将位姿时间戳排序，便于快速查找
    sorted_pose_ts = sorted(poses_w2c_by_ts.keys())

    # --- 修改：创建列表来收集动态增长的位姿 ---
    dynamic_predicted_poses = []
    dynamic_gt_poses = []

    bag_path = args.output_bag
    with rosbag.Bag(bag_path, 'w') as bag:
        print(f"开始处理帧并写入动态数据到 ROS Bag: {bag_path}")

        # --- 写入一次性的、完整的地图点云 ---
        print("正在写入主地图点云...")
        map_pts_xyz = np.array([p.xyz for p in points3D.values()])
        map_pts_rgb = np.array([p.rgb for p in points3D.values()]) / 255.0
        map_pts_full = np.hstack([map_pts_xyz, map_pts_rgb])
        # 使用一个早于所有数据的时间戳，确保它在 bag 文件最前面
        map_stamp = rospy.Time(0) 
        map_msg = to_pointcloud2_msg(map_pts_full, map_stamp)
        bag.write('/map_pointcloud', map_msg, t=map_stamp)

        sorted_queries = sorted(query_data.items(), key=lambda item: item[1][0])

        for h5_full_key, (query_ts, query_filename) in tqdm(sorted_queries, desc="处理帧"):
            # ... (查找最近位姿的逻辑不变) ...
            import bisect
            idx = bisect.bisect_left(sorted_pose_ts, query_ts)
            if not sorted_pose_ts: break
            if idx == 0: best_ts = sorted_pose_ts[0]
            elif idx == len(sorted_pose_ts): best_ts = sorted_pose_ts[-1]
            else:
                before, after = sorted_pose_ts[idx - 1], sorted_pose_ts[idx]
                best_ts = before if query_ts - before < after - query_ts else after
            if abs(query_ts - best_ts) > 0.1: continue

            qvec, tvec = poses_w2c_by_ts[best_ts]
            stamp = rospy.Time.from_sec(query_ts)

            # --- 写入瞬时数据 (图像, TF, 内点) ---
            # a. 写入图像
            image_path = args.query_dir / query_filename
            if image_path.exists():
                # 使用 OpenCV 读取图像
                cv_image = cv2.imread(str(image_path))
                if cv_image is not None:
                    # 使用 CvBridge 将 OpenCV 图像转换为 ROS Image 消息
                    # 'bgr8' 是 cv2.imread 的默认颜色顺序
                    image_msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                    image_msg.header.stamp = stamp
                    image_msg.header.frame_id = "camera" # 图像通常在相机坐标系下
                    bag.write('/camera/image_raw', image_msg, t=stamp)
            else:
                print(f"警告: 找不到查询图像文件: {image_path}")

            # --- 核心修改：在每一帧都写入增长的 Path 消息 ---
            # 1. 处理预测位姿
            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = tvec
            pose_msg.pose.orientation.w, pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z = qvec
            
            # 将当前位姿加入列表
            dynamic_predicted_poses.append(pose_msg)
            
            # 创建并写入“到目前为止”的路径
            path_msg = RosPath()
            path_msg.header.stamp = stamp
            path_msg.header.frame_id = "map"
            path_msg.poses = dynamic_predicted_poses
            bag.write('/camera_path', path_msg, t=stamp) # 写入动态路径

            # 写入瞬时 TF
            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = "map"
            tf_msg.child_frame_id = "camera"
            tf_msg.transform.translation.x, tf_msg.transform.translation.y, tf_msg.transform.translation.z = tvec
            tf_msg.transform.rotation.x, tf_msg.transform.rotation.y, tf_msg.transform.rotation.z, tf_msg.transform.rotation.w = qvec[1], qvec[2], qvec[3], qvec[0]
            bag.write('/tf', TFMessage([tf_msg]), t=stamp)

            # 2. (可选) 处理真值位姿
            if ground_truth_poses_map:
                # 找到与当前查询时间戳最匹配的真值位姿
                gt_timestamps = list(ground_truth_poses_map.keys())
                gt_idx = bisect.bisect_left(gt_timestamps, query_ts)
                if gt_idx > 0:
                    best_gt_ts = gt_timestamps[gt_idx - 1]
                    gt_pose_msg = ground_truth_poses_map[best_gt_ts]
                    
                    dynamic_gt_poses.append(gt_pose_msg)
                    
                    gt_path_msg = RosPath()
                    gt_path_msg.header.stamp = stamp
                    gt_path_msg.header.frame_id = "map"
                    gt_path_msg.poses = dynamic_gt_poses
                    bag.write('/ground_truth_path', gt_path_msg, t=stamp)

            # c. 写入匹配的3D点
            if h5_full_key in match_file:
                print(f"处理匹配文件中的键: {h5_full_key}")
                try:
                    for cam_name in match_file[h5_full_key].keys():
                        print(f"处理相机组: {cam_name}")
                        cam_group = match_file[h5_full_key][cam_name]

                        best_ref_image_name = None  
                        best_score = -1

                        for ref_image_name in cam_group.keys():
                            print(f"处理参考图像: {ref_image_name}")
                            ref_group = cam_group[ref_image_name]

                            if 'matches0' in ref_group and 'matching_scores0' in ref_group:
                                print(f"找到匹配数据集: matches0 和 matching_scores0")
                                scores = ref_group['matching_scores0'].__array__()
                                avg_score = np.mean(scores)
                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_ref_image_name = ref_image_name
                            else:
                                print(f"警告: 缺少匹配数据集，参考图像: {ref_image_name}")

                        if best_ref_image_name:
                            print(f"得分最高的参考图像: {best_ref_image_name}")
                            ref_group = cam_group[best_ref_image_name]
                            matches = ref_group['matches0'].__array__()
                            scores = ref_group['matching_scores0'].__array__()
                            inliers = scores > 0.2
                            print(f"内点数量: {np.sum(inliers) if inliers is not None else 0}")

                            if np.sum(inliers) == 0:
                                print(f"警告: 没有有效的内点，查询图像: {h5_full_key}, 参考图像: {best_ref_image_name}")
                                continue

                    for h5_full_key, (query_ts, query_filename) in tqdm(sorted_queries, desc="处理帧"):
                                # 从 logs 中获取 PnP 求解结果
                                loc = logs["loc"].get(h5_full_key)  # 替换为查询图像的 key
                                if loc is None:
                                    print(f"警告: 查询图像 {h5_full_key} 不在 logs 中。")
                                    continue

                                PnP_ret = loc.get("PnP_ret")  # 提取 PnP 求解的结果
                                if PnP_ret is not None and "inliers" in PnP_ret:
                                    inliers = np.array(PnP_ret["inliers"])  # 提取内点布尔数组
                                else:
                                    print(f"警告: 查询图像 {h5_full_key} 的 PnP 求解结果无效或缺少内点信息。")
                                    inliers = np.zeros(len(matches), dtype=bool)  # 如果没有内点信息，默认全为外点

                                print(f"内点数量: {np.sum(inliers) if inliers is not None else 0}")

                                if np.sum(inliers) == 0:
                                    print(f"警告: 没有有效的内点，查询图像: {h5_full_key}")
                                    continue

                                # 加载查询图像和参考图像
                                query_image = cv2.imread(str(args.query_dir / h5_full_key))
                                ref_image = cv2.imread(str(args.db_image_dir / cam_name / ref_image_name))

                                if query_image is None:
                                    print(f"警告: 查询图像加载失败: {args.query_dir / h5_full_key}")
                                    continue
                                if ref_image is None:
                                    print(f"警告: 参考图像加载失败: {args.db_image_dir / cam_name / ref_image_name}")
                                    continue

                                # 从查询图像特征文件中加载关键点
                                kp_query = query_feature_file[h5_full_key]['keypoints'].__array__()
                                kp_ref = ref_feature_file[cam_name][ref_image_name]['keypoints'].__array__()

                                # 调用可视化函数
                                vis_image = save_visualization_as_image(
                                    query_image, ref_image, kp_query, kp_ref, inliers, h5_full_key, ref_image_name
                                )
                                if vis_image is None:
                                    print(f"警告: 可视化图像生成失败，查询图像: {h5_full_key}, 参考图像: {ref_image_name}")
                                    continue

                                # 将可视化图像保存为 ROS Image 消息
                                vis_msg = bridge.cv2_to_imgmsg(vis_image, "bgr8")
                                vis_msg.header.stamp = rospy.Time.from_sec(query_ts)
                                vis_msg.header.frame_id = "map"
                                bag.write('/visualization_image', vis_msg, t=rospy.Time.from_sec(query_ts))
                                print(f"成功写入可视化图像到话题 /visualization_image，查询图像: {h5_full_key}, 参考图像: {ref_image_name}")

                except Exception as e:
                    print(f"警告: 写入可视化图像时发生错误: {e}")
                    continue


    print(f"ROS Bag 创建完成: {bag_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 hloc 结果创建 ROS Bag")
    
    # --- 修改: 将所有 Path 类型改为 str 类型，以便手动处理 ---
    parser.add_argument("--sfm_dir", type=str, required=True, help="包含重建模型的目录路径")
    parser.add_argument("--query_dir", type=str, required=True, help="包含查询图像的目录路径")
    parser.add_argument("--results_file", type=str, required=True, help="包含定位结果的 .txt 文件路径")
    parser.add_argument("--ground_truth_file", type=str, help="包含真值位姿的文件路径")
    parser.add_argument("--query_feature_file", type=str, required=True, help="查询图像特征文件路径")
    parser.add_argument("--ref_feature_file", type=str, required=True, help="参考图像的特征文件路径")
    parser.add_argument("--match_file", type=str, required=True, help="HDF5 匹配文件路径")
    parser.add_argument("--output_bag", type=str, default="hloc_visualization.bag", help="输出的 ROS Bag 文件名")
    parser.add_argument("--db_image_dir", type=str, required=True, help="数据库图像的目录路径")
    # 添加 logs 文件路径参数
    parser.add_argument("--logs_file", type=str, required=True, help="包含定位结果的 logs.pkl 文件路径")
    
    
    args = parser.parse_args()

    # --- 新增: 手动将字符串转换为 Path 对象 ---
    args.sfm_dir = Path(args.sfm_dir)
    args.query_dir = Path(args.query_dir)
    args.results_file = Path(args.results_file)
    args.query_feature_file = Path(args.query_feature_file)
    args.ref_feature_file = Path(args.ref_feature_file)
    args.match_file = Path(args.match_file)
    args.db_image_dir = Path(args.db_image_dir)  # 确保 db_image_dir 是 Path 对象
    args.logs_file = Path(args.logs_file)  # 添加 logs 文件路径

    # 检查关键目录和文件是否存在
    if not args.sfm_dir.is_dir():
        print(f"错误: --sfm_dir 指定的路径不是一个有效的目录: {args.sfm_dir}")
        exit(1)
    if not args.query_dir.is_dir():
        print(f"错误: --query_dir 指定的路径不是一个有效的目录: {args.query_dir}")
        exit(1)
    if not args.results_file.is_file():
        print(f"错误: --results_file 指定的文件不存在: {args.results_file}")
        exit(1)
    if not args.logs_file.is_file():
        print(f"错误: --logs_file 指定的文件不存在: {args.logs_file}")
        exit(1)

    # 对于可选的真值文件，也进行转换和检查
    if args.ground_truth_file:
        args.ground_truth_file = Path(args.ground_truth_file)
        if not args.ground_truth_file.is_file():
            print(f"错误: --ground_truth_file 指定的文件不存在: {args.ground_truth_file}")
            exit(1)

    # 所有检查通过后，调用 main 函数
    main(args)
