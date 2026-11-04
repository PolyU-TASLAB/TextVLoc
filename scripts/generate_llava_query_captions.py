"""Paired, resumable LLaVA captions; training and baseline captions stay unchanged."""
import argparse
import csv
from pathlib import Path

from scripts.generate_query_prompt_variants import digest, read_queries, prepare_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--baseline-csv', type=Path, default=Path('outputs/SensorKit_hloc/allcams_spatial_colmap_0630_CAM1_0.025/qwen3_query_prompts_qwen-vl_77clip10.csv'))
    parser.add_argument('--query-dir', type=Path, default=Path('datasets/SensorKit/query/cam1_all/0_undistorted'))
    parser.add_argument('--output-csv', type=Path, default=Path('outputs/query_text_generalization/llava_mistral7b_qwen_prompt_v3/original_prompt.csv'))
    args = parser.parse_args()
    baseline, root, output = args.baseline_csv, args.query_dir, args.output_csv
    from hloc.generate_captions import QWEN_VL_NAV_PROMPT, get_captioner
    prompt = QWEN_VL_NAV_PROMPT
    names = read_queries(baseline, root)
    model_id = 'llava-hf/llava-v1.6-mistral-7b-hf'
    revision = '2424fdd47412fccc66d91719126b420e9fbd7065'
    manifest = dict(model_id=model_id, revision=revision, prompt=prompt,
                    baseline_sha256=digest(baseline), query_names=names,
                    query_dir=str(root.resolve()), text_source='query_image_only',
                    human_written=False, dtype='float16',
                    generation=dict(max_new_tokens=96, do_sample=False),
                    runner_sha256=digest(Path(__file__)),
                    captioner_sha256=digest(Path('hloc/generate_captions.py')))
    pending = prepare_run(output, manifest, names, args.dry_run)
    print(f'total={len(names)} completed={len(names)-len(pending)} pending={len(pending)}', flush=True)
    if args.dry_run or not pending:
        return
    import torch
    import transformers
    print(f'Loading {model_id}@{revision}; torch={torch.__version__} transformers={transformers.__version__}', flush=True)
    caption = get_captioner('llava', 'cuda', revision=revision, load_in_4bit=False)
    exists = output.exists()
    with output.open('a' if exists else 'x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(['image_name', 'description'])
            handle.flush()
        for index, name in enumerate(pending, 1):
            description = caption(root / name, prompt=prompt, max_new_tokens=96)
            if not description:
                raise RuntimeError(f'Empty caption: {name}; stopping without dropping query')
            writer.writerow([name, description])
            handle.flush()
            print(f'{len(names)-len(pending)+index}/{len(names)} {name}: {description}', flush=True)
    print('COMPLETE: all query captions saved', flush=True)


if __name__ == '__main__':
    main()
