import time
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 导入 hloc 的核心模块
from hloc import extract_features, pairs_from_retrieval, match_features, localize_sfm
from hloc.utils.parsers import parse_image_lists


def profile_single_image(query_name, query_dir, outputs, args):
    """对单张图像进行定位并分析耗时"""
    
    # --- 配置 ---
    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_max"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    
    reference_sfm = Path(args.reference_sfm)
    
    # --- 计时开始 ---
    total_start_time = time.time()
    timings = {}

    # 步骤 1: 提取全局特征
    start_t = time.time()
    # 注意：hloc设计为批处理，这里我们模拟单张处理
    # 实际中它会检查文件是否存在，如果存在则跳过，所以我们先确保它不存在
    (outputs / retrieval_conf['output']).with_suffix('.h5').unlink(missing_ok=True)
    global_descriptors = extract_features.main(retrieval_conf, query_dir, outputs, image_list=[query_name])
    timings['global_feature'] = time.time() - start_t

    # 步骤 2: 图像检索
    start_t = time.time()
    loc_pairs_path = outputs / f"pairs_for_{Path(query_name).stem}.txt"
    pairs_from_retrieval.main(
        global_descriptors, loc_pairs_path, args.num_loc, db_model=reference_sfm, query_list=[query_name]
    )
    timings['retrieval'] = time.time() - start_t

    # 步骤 3: 提取局部特征
    start_t = time.time()
    (outputs / feature_conf['output']).with_suffix('.h5').unlink(missing_ok=True)
    features = extract_features.main(feature_conf, query_dir, outputs, image_list=[query_name], as_half=True)
    timings['local_feature'] = time.time() - start_t

    # 步骤 4: 局部特征匹配
    start_t = time.time()
    loc_matches_path = outputs / matcher_conf['output']
    loc_matches_path = loc_matches_path.with_suffix('.h5')
    loc_matches_path.unlink(missing_ok=True)
    loc_matches = match_features.main(matcher_conf, loc_pairs_path, feature_conf["output"], outputs)
    timings['matching'] = time.time() - start_t

    # 步骤 5: 位姿估计
    start_t = time.time()
    results_path = outputs / f"results_for_{Path(query_name).stem}.txt"
    localize_sfm.main(
        reference_sfm, [query_name], loc_pairs_path, features, loc_matches, results_path,
        covisibility_clustering=False # 单帧处理不需要聚类
    )
    timings['pose_estimation'] = time.time() - start_t
    
    timings['total'] = time.time() - total_start_time
    return timings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile single-frame localization time.")
    parser.add_argument("--query_dir", type=Path, required=True, help="Path to the query images directory.")
    parser.add_argument("--reference_sfm", type=Path, required=True, help="Path to the reference SfM model.")
    parser.add_argument("--outputs", type=Path, required=True, help="Path to the output directory for features and matches.")
    parser.add_argument("--num_loc", type=int, default=20, help="Number of image pairs for localization.")
    args = parser.parse_args()

    query_images = [p.name for p in sorted(list(args.query_dir.glob("*.jpg")))]
    if not query_images:
        raise ValueError(f"No JPG images found in {args.query_dir}")

    all_timings = []
    for query_name in tqdm(query_images, desc="Profiling images"):
        timings = profile_single_image(query_name, args.query_dir, args.outputs, args)
        timings['image'] = query_name
        all_timings.append(timings)

    # 打印结果
    df = pd.DataFrame(all_timings)
    print("\n--- Localization Profiling Results (seconds) ---")
    print(df.to_string(index=False))
    
    print("\n--- Average Timings (seconds) ---")
    print(df.mean(numeric_only=True))

