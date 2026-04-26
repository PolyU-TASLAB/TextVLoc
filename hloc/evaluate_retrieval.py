import argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from PIL import Image
import cv2

try:
    from transformers import CLIPProcessor, CLIPModel
    # ✅ 1. 引入 loralib 库
    import loralib as lora
except ImportError:
    # ✅ 修复这里的错误提示
    raise ImportError("Please install transformers and loralib: pip install transformers loralib")



class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _forward_hook(self, module, input, output):
        self.activations = output[0]

    def __call__(self, pixel_values, input_ids):
        self.model.zero_grad()
        outputs = self.model(pixel_values=pixel_values, input_ids=input_ids)
        similarity = outputs.logits_per_image.squeeze()
        similarity.backward()
        pooled_gradients = torch.mean(self.gradients, dim=[0, 1])
        activations = self.activations.detach()
        # Check dimension
        # if activations.dim() == 2:
        #     return None
        for i in range(pooled_gradients.shape[-1]):
            activations[:, :, i] *= pooled_gradients[..., i]
        heatmap = torch.mean(activations, dim=-1).squeeze()
        # Remove CLS token
        heatmap = heatmap[1:]
        side_len = int(np.sqrt(heatmap.shape[0]))
        heatmap = heatmap.reshape(side_len, side_len).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        return heatmap

def load_pairs(path: Path) -> defaultdict:
    pairs = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split()
            if len(parts) == 2:
                q, db = parts
                pairs[q].append(db)
    return pairs


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model {args.model_id} on {device}...")
    model = CLIPModel.from_pretrained(args.model_id).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_id)

# ============================================================
    # ✅ 3. 修改模型加载逻辑，应用 loralib 权重
    # ============================================================
    if args.lora_ckpt_path:
        print(f"Applying LoRA weights from: {args.lora_ckpt_path}")
        try:
            # 步骤 1: 遍历模型，将 nn.Linear 替换为 lora.Linear
            # 注意：这里假设LoRA被应用于所有的Q, K, V投影矩阵，这与常见的CLIP LoRA微调一致
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    # 获取原始层的参数
                    in_features = module.in_features
                    out_features = module.out_features
                    # 创建新的 lora.Linear 层
                    # r=8, lora_alpha=8 是常见默认值，如果您的训练脚本不同，请修改
                    new_layer = lora.Linear(in_features, out_features, r=8, lora_alpha=8, bias=module.bias is not None)
                    # 复制原始权重
                    new_layer.weight.data = module.weight.data
                    if module.bias is not None:
                        new_layer.bias.data = module.bias.data
                    
                    # 找到父模块并替换子模块
                    # 这段代码比较复杂，用于动态替换模型中的层
                    parent_name = ""
                    for name, sub_module in model.named_modules():
                        if module is sub_module:
                            parent_name = name
                            break
                    if parent_name:
                        parent_module = model.get_submodule(parent_name.rsplit('.', 1)[0])
                        setattr(parent_module, parent_name.rsplit('.', 1)[-1], new_layer.to(device))

            print("  > Model layers converted to lora.Linear.")

            # 步骤 2: 加载 LoRA 权重
            lora_weights = torch.load(args.lora_ckpt_path, map_location=device)
            # 使用 strict=False 加载，因为它只包含LoRA参数
            model.load_state_dict(lora_weights, strict=False)
            print("✅ LoRA weights applied successfully using loralib.")

        except Exception as e:
            print(f"❌ Error applying LoRA weights with loralib: {e}")
            print("   Proceeding with the base model only.")
    # ============================================================

    model.eval()

    print("Loading data files...")
    retrieved_pairs = load_pairs(args.retrieval_pairs)
    retrieved_pairs = {Path(q).name: dbs for q, dbs in retrieved_pairs.items()}

    prompts_df = pd.read_csv(args.prompts_csv, index_col='image_name')
    prompts_df.index = prompts_df.index.str.strip().str.strip('"')
    prompts_df.index = prompts_df.index.map(lambda x: Path(x).name)

    # Debug name matching
    retrieval_keys = list(retrieved_pairs.keys())
    prompt_keys = list(prompts_df.index)
    common_keys = set(retrieval_keys) & set(prompt_keys)
    print(f"Found {len(common_keys)} common query names out of {len(retrieval_keys)}.")

    grad_cam = None
    try:
        # target_layer = model.vision_model.encoder.layers[-1].layer_norm2

        target_layer = model.vision_model.encoder.layers[-1].self_attn

        # target_layer = model.vision_model.encoder.layers[-1]

        grad_cam = GradCAM(model, target_layer)
        print("Grad-CAM initialized successfully.")
    except AttributeError:
        print("Warning: Could not find target layer for Grad-CAM. Heatmap generation will be skipped.")

    results = []
    output_viz_dir = args.output_dir / "grad_cam_viz"
    output_viz_dir.mkdir(exist_ok=True, parents=True)

    for q_name, db_names in tqdm(retrieved_pairs.items(), desc="Evaluating retrieval"):
        if q_name not in prompts_df.index:
            continue
        prompt = prompts_df.loc[q_name, 'description']
        q_img_path = args.image_dir / q_name
        if not q_img_path.exists():
            print(f"[MISS] Image file {q_img_path} does not exist.")
            continue

        image = Image.open(q_img_path).convert("RGB")
        inputs = processor(
            text=[prompt], images=image,
            return_tensors="pt", padding=True,
            truncation=True, max_length=77
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # cosine_sim = outputs.logits_per_image.detach().cpu().item()
            cosine_sim = (outputs.logits_per_image / model.logit_scale.exp()).detach().cpu().item()


        results.append({'query': q_name, 'cosine_sim': cosine_sim})

        # Grad-CAM visualization
        if grad_cam:
            heatmap = grad_cam(pixel_values=inputs['pixel_values'], input_ids=inputs['input_ids'])
            if heatmap is not None:
                try:
                    img_cv = cv2.imread(str(q_img_path))
                    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    superimposed_img = cv2.addWeighted(heatmap_colored, 0.5, img_cv, 0.5, 0)
                    cv2.imwrite(str(output_viz_dir / f"{q_name}_gradcam.jpg"), superimposed_img)
                except Exception as e:
                    print(f"[Grad-CAM Error during visualization for {q_name}: {e}]")

    if not results:
        print("No valid results produced. Please check input files and prompts.")
        return

    results_df = pd.DataFrame(results)
    avg_cosine = results_df['cosine_sim'].mean()
    print("\n--- Retrieval Evaluation Results ---")
    print(f"Average Cosine Similarity: {avg_cosine:.4f}")

    output_csv_path = args.output_dir / "evaluation_scores.csv"
    results_df.to_csv(output_csv_path, index=False)
    print(f"Detailed scores saved to: {output_csv_path}")
    if grad_cam:
        print(f"Grad-CAM visualizations saved to: {output_viz_dir}")
    print("------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLIP-based retrieval with Cosine Similarity and Grad-CAM.")
    parser.add_argument("--retrieval_pairs", type=Path, required=True)
    parser.add_argument("--prompts_csv", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_id", type=str, default='laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', help="Base CLIP model ID from Hugging Face.")
    
    # ============================================================
    # ✅ 2. 添加新的命令行参数
    # ============================================================
    parser.add_argument("--lora_ckpt_path", type=Path, default=None, help="Optional path to the LoRA checkpoint directory.")
    # ============================================================

    args = parser.parse_args()
    main(args)




    # python3 hloc/evaluate_retrieval.py \
    # --retrieval_pairs outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/pairs-for-matching-qwen-vl.txt \
    # --prompts_csv outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/qwen3_query_prompts_qwen-vl_77clip10.csv \
    # --image_dir datasets/SensorKit/query/cam1_all/0_undistorted/ \
    # --output_dir outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/evaluation_qwen-vl \
    # --lora_ckpt_path datasets/SensorKit/clipdata/finetune_output/stg2_00_qwen3_clipdata_text_mask_prepare_sparse03_remove/lora_epoch_40.pt






