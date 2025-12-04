# import matplotlib.pyplot as plt
# from PIL import Image
# from pathlib import Path
# import collections

# # 根目录，包含 cam0, cam1, ..., cam6 文件夹
# root_dir = Path("datasets/SensorKit/mapping_allcams_downsampled_10_10")  
# query_dir = Path("datasets/SensorKit/query_allcams_downsampled_10_10")  
# # query_dir = root_dir / "cam1"  # 查询图像目录
# pairs_file = Path("outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/downsampled10_10_pairs-query-netvlad10.txt")
# output_dir = Path("downsample10_10_retrieval_netvlad")
# output_dir.mkdir(exist_ok=True)

# # 1. 读取 pairs.txt 并按查询图像分组
# grouped = collections.defaultdict(list)
# with pairs_file.open("r") as f:
#     for line in f:
#         q, r = line.strip().split()
#         grouped[q].append(r)

# # 2. 为每个查询图像生成可视化（示例仅展示第一组）
# for i, (query, retrieved_list) in enumerate(grouped.items()):
#     fig = plt.figure(figsize=(18, 10))
#     gs = fig.add_gridspec(4, 6, width_ratios=[1] + [1]*5)

#     # 左列：查询图像从 cam1 文件夹加载
#     ax_q = fig.add_subplot(gs[:, 0])
#     img_q = Image.open(query_dir / query)
#     ax_q.imshow(img_q)
#     ax_q.set_title("query_image", fontsize=14)
#     ax_q.axis('off')


#     # 右侧网格：前 20 张检索结果，从所有 camX 文件夹加载
#     for idx, ret in enumerate(retrieved_list[:20]):
#         row = idx // 5
#         col = idx % 5
#         ax = fig.add_subplot(gs[row, col + 1])
#         img_r = Image.open(root_dir / ret)
#         ax.imshow(img_r)
#         ax.set_title(f"Rank {idx+1}", fontsize=10)
#         ax.axis('off')

#     plt.tight_layout()
#     save_path = output_dir / f"{Path(query).stem}_retrieval.png"
#     plt.savefig(save_path, dpi=150)
#     # plt.show()
#     plt.close(fig)

#     # # 仅示例展示第一组，可移除此条件展示所有
#     # if i == 0:
#     #     break








import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import collections
import sys

def visualize_retrieval(db_root, query_root, pairs_path, output_path, num_to_show=20):
    """
    为图像检索结果生成可视化图像。

    Args:
        db_root (Path): 数据库图像的根目录。
        query_root (Path): 查询图像的根目录。
        pairs_path (Path): hloc 生成的图像对文件路径。
        output_path (Path): 保存可视化结果的目录。
        num_to_show (int): 每个查询最多显示的检索结果数量。
    """
    if not db_root.exists():
        print(f"错误：数据库根目录不存在: {db_root}", file=sys.stderr)
        return
    if not query_root.exists():
        print(f"错误：查询根目录不存在: {query_root}", file=sys.stderr)
        return
    if not pairs_path.exists():
        print(f"错误：图像对文件不存在: {pairs_path}", file=sys.stderr)
        return

    output_path.mkdir(exist_ok=True)

    # 1. 读取并分组图像对
    grouped = collections.defaultdict(list)
    with pairs_path.open("r") as f:
        for line in f:
            # try:
            #     q, r = line.strip().split()
            #     grouped[q].append(r)
            # except ValueError:
            #     continue  # 跳过格式不正确的行
            # try:
            #     parts = line.strip().split()
            #     if len(parts) >= 2:
            #         q, r = parts[0], parts[1]
            #         grouped[q].append(r)
            # except ValueError:
            #     continue
            try:
                parts = line.strip().split()
                if len(parts) >= 2:
                    query_image = parts[0]
                    # 检索到的图像路径在索引 1, 3, 5, ...
                    # 我们用步长为2的循环来提取它们
                    retrieved_images = [parts[i] for i in range(1, len(parts), 2)]
                    grouped[query_image].extend(retrieved_images)
            except (ValueError, IndexError):
                # 跳过空行或格式不正确的行
                continue

    if not grouped:
        print("警告：图像对文件为空或格式不正确，未生成任何可视化。")
        return

    total_queries = len(grouped)
    print(f"找到 {total_queries} 个查询图像，开始生成可视化...")

    # 2. 为每个查询生成可视化
    for i, (query_rel_path, retrieved_rel_paths) in enumerate(grouped.items()):
        fig = plt.figure(figsize=(20, 10))
        # 动态调整网格布局以适应 num_to_show
        cols = 5
        rows = (num_to_show + cols - 1) // cols
        gs = fig.add_gridspec(rows, cols + 1, width_ratios=[2] + [1]*cols, wspace=0.1, hspace=0.3)

        # 左列：显示查询图像
        ax_q = fig.add_subplot(gs[:, 0])
        query_full_path = query_root / query_rel_path
        try:
            img_q = Image.open(query_full_path)
            ax_q.imshow(img_q)
            ax_q.set_title(f"Query:\n{query_rel_path}", fontsize=10)
        except FileNotFoundError:
            ax_q.text(0.5, 0.5, f"Image not found:\n{query_full_path}", ha='center', va='center', wrap=True, color='red')
        ax_q.axis('off')

        # 右侧网格：显示检索到的数据库图像
        for idx, ret_rel_path in enumerate(retrieved_rel_paths[:num_to_show]):
            row = idx // cols
            col = idx % cols
            ax = fig.add_subplot(gs[row, col + 1])
            db_full_path = db_root / ret_rel_path
            try:
                img_r = Image.open(db_full_path)
                ax.imshow(img_r)
                ax.set_title(f"Rank {idx+1}", fontsize=8)
            except FileNotFoundError:
                ax.text(0.5, 0.5, "Not Found", ha='center', va='center', color='red')
            ax.axis('off')

        # 保存图像
        # 使用查询图像的相对路径来创建唯一的文件名，替换斜杠以避免创建子目录
        safe_filename = query_rel_path.replace('/', '_')
        save_path = output_path / f"{Path(safe_filename).stem}_retrieval.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if (i + 1) % 10 == 0 or (i + 1) == total_queries:
            print(f"  ...已处理 {i+1}/{total_queries}")

    print(f"可视化完成！结果已保存到: {output_path}")


if __name__ == "__main__":
    # --- 配置区域 ---
    # !! 请根据您的设置仔细检查以下路径 !!

    # 数据库图像（被检索到的图像）的根目录
    database_root = Path("datasets/SensorKit/allcams")  

    # 查询图像的根目录
    query_root = Path("datasets/SensorKit/query/cam1_all/0_undistorted")  

    # hloc 生成的图像对文件
    pairs_file = Path("outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/pairs-query-clip5-qwen-vl.txt")
    
    # 保存可视化结果的目录
    output_dir = Path("pairs-query-clip5-qwen-vl_dampclip")

    # --- 运行 ---
    visualize_retrieval(
        db_root=database_root,
        query_root=query_root,
        pairs_path=pairs_file,
        output_path=output_dir,
        num_to_show=5  # 您想为每个查询显示多少个检索结果
    )