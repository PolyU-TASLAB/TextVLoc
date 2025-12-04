import argparse
import glob
import logging
from pathlib import Path

from ... import (
    extract_features,
    localize_sfm,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)
from . import colmap_from_nvm

CONDITIONS = [
    "dawn",
    "dusk",
    "night",
    "night-rain",
    "overcast-summer",
    "overcast-winter",
    "rain",
    "snow",
    "sun",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 
def generate_query_list(dataset, image_dir, path):
    h, w = 1024, 1024
    intrinsics_filename = "intrinsics/{}_intrinsics.txt"
    cameras = {}
    for side in ["left", "right", "rear"]:
        with open(dataset / intrinsics_filename.format(side), "r") as f:
            fx = f.readline().split()[1]
            fy = f.readline().split()[1]
            cx = f.readline().split()[1]
            cy = f.readline().split()[1]
            assert fx == fy
            params = ["SIMPLE_RADIAL", w, h, fx, cx, cy, 0.0]
            cameras[side] = [str(p) for p in params]

    queries = glob.glob((image_dir / "**/*.jpg").as_posix(), recursive=True)
    queries = [
        Path(q).relative_to(image_dir.parents[0]).as_posix() for q in sorted(queries)
    ]

    out = [[q] + cameras[Path(q).parent.name] for q in queries]
    with open(path, "w") as f:
        f.write("\n".join(map(" ".join, out)))


def run(args):
    print("Starting RobotCar pipeline...")
    logger.info("Starting pipeline...")
    # Setup the paths
    dataset = args.dataset
    images = dataset / "images/"
    logger.info(f"Dataset path: {dataset}")
    logger.info(f"Images path: {images}")

    outputs = args.outputs  # where everything will be saved
    outputs.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {outputs}")

    # 每种条件(天气/光照)下的查询图像列表文件，包含相机内参
    # 格式: 图像路径 相机模型 宽度 高度 焦距 cx cy 径向畸变
    # 可视化: 可使用COLMAP的GUI查看这些图像及其内参
    query_list = outputs / "{condition}_queries_with_intrinsics.txt"
    
    # SIFT特征提取和匹配构建的SfM(Structure from Motion)模型
    # SIFT是一种经典且稳定的特征提取算法，具有良好的尺度和旋转不变性
    # 这里用于初始三维重建是因为其可靠性和成熟度高，为后续更精确的重建奠定基础
    # NVM(N-View Match)模型是VisualSFM软件的输出格式，包含相机参数、特征点和3D点云
    # 这里从已有的NVM模型转换为COLMAP格式，以利用COLMAP的现代化工具链
    # 可视化: 使用COLMAP GUI打开数据库或使用COLMAP命令行导出为PLY点云
    sift_sfm = outputs / "sfm_sift"
    
    # 使用SuperPoint特征和SuperGlue匹配器构建的SfM模型
    # 这是最终用于定位的参考模型，精度高于SIFT
    # 可视化: 同样使用COLMAP GUI或导出为PLY点云后用MeshLab/CloudCompare查看
    reference_sfm = outputs / "sfm_superpoint+superglue"
    
    # 基于共视性生成的图像对文件，用于SfM重建
    # 格式: 每行两个图像ID，表示它们共享视野
    # 可视化: 可使用networkx库将其可视化为图形网络
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    
    # 基于NetVLAD检索的图像对文件，用于定位
    # 格式: 每行为查询图像和检索到的最相似参考图像
    # 可视化: 可编写脚本并排显示查询图像和检索图像
    loc_pairs = outputs / f"pairs-query-netvlad{args.num_loc}.txt"
    
    # 最终定位结果文件，包含每个查询图像的位姿估计
    # 格式: 图像名 qw qx qy qz tx ty tz 
    # 可视化: 可使用Open3D或Matplotlib绘制相机位置和方向，或与地面真值对比
    results = outputs / f"RobotCar_hloc_superpoint+superglue_netvlad{args.num_loc}.txt"

    # # 如果只运行定位步骤
    # if args.step == "localization":
    #     features = outputs / "feats-superpoint-n4096-r1024.h5"
    #     loc_matches = outputs / "feats-superpoint-n4096-r1024_matches-superglue_pairs-query-netvlad20.h5"
    #     query_list = outputs / "*_queries_with_intrinsics.txt"  # 通配符匹配所有条件
    #     if not features.exists():
    #         raise FileNotFoundError(f"Features file not found: {features}")
    #     if not loc_matches.exists():
    #         raise FileNotFoundError(f"Matches file not found: {loc_matches}")
    #     if not reference_sfm.exists():
    #         raise FileNotFoundError(f"Reference SfM model not found: {reference_sfm}")
    #     localize_sfm.main(
    #         reference_sfm,
    #         Path(str(query_list).format(condition="*")),
    #         loc_pairs,
    #         features,
    #         loc_matches,
    #         results,
    #         covisibility_clustering=False,
    #         prepend_camera_name=True,
    #     )
    #     return



    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]

    for condition in CONDITIONS:
        generate_query_list(
            dataset, images / condition, str(query_list).format(condition=condition)
        )

    features = extract_features.main(feature_conf, images, outputs, as_half=True)

    colmap_from_nvm.main(
        dataset / "3D-models/all-merged/all.nvm",
        dataset / "3D-models/overcast-reference.db",
        sift_sfm,
    )
    pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    triangulation.main(
        reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches
    )

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    # TODO: do per location and per camera
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_prefix=CONDITIONS,
        db_model=reference_sfm,
    )
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )

    localize_sfm.main(
        reference_sfm,
        Path(str(query_list).format(condition="*")),
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
        prepend_camera_name=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        help="Specify a single step to run (e.g., 'localization'). Default: run all steps.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/robotcar",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/robotcar",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=20,
        help="Number of image pairs for loc, default: %(default)s",
    )
    args = parser.parse_args()
    run(args)
