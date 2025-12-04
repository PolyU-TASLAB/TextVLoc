import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path
import torch
from PIL import Image
import csv
from tqdm import tqdm
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig,
    AutoProcessor, LlavaOnevisionForConditionalGeneration, 
    AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)

from . import logger

def get_captioner(model_name: str, device: str):
    """
    工厂函数：根据模型名称加载并返回一个 captioning 函数。
    这可以避免在循环中重复加载模型。
    """
    logger.info(f"Loading model '{model_name}' onto device '{device}'...")

    if model_name == 'blip':
        # processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        # model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            torch_dtype=torch.float16
        ).to(device).eval()


        # processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        # model = Blip2ForConditionalGeneration.from_pretrained(
        #     "Salesforce/blip2-opt-2.7b",
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        # )

        def caption(image_path: Path) -> str:
            image = Image.open(image_path).convert("RGB")
            
            # 1. 正确的调用方式：这个模型不需要 text 参数
            inputs = processor(image, return_tensors="pt").to(device)
            
            # 2. 生成描述
            out = model.generate(**inputs, max_new_tokens=75) # 对于简单描述，75个token足够
            
            # 3. 解码并返回结果
            return processor.decode(out[0], skip_special_tokens=True).strip()
        return caption

    elif model_name == 'llava':
        # model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    

        prompt_template = """USER: <image>
        You are a navigation assistant observing this image.
        Describe the scene as if you're standing at the camera’s viewpoint.
        Use first-person perspective (“I’m at...”, “On my left...”, etc.).
        Focus on:
        - Unique or permanent landmarks (e.g., logos, building names, statues)
        - Directional relationships (e.g., “on my right”, “ahead”, “behind me”)
        - Road structure or intersections
        - Building materials or signs that help visual recognition

        Keep the description concise and natural. Avoid speculation or subjective opinions.
        ASSISTANT:"""



        # prompt_template = """USER: <image>
        # You are a vision tagger for CLIP retrieval.

        # Task: Output exactly 6 short English phrases (3–6 words each) that uniquely identify this place using only static, long-lived elements.

        # Focus on: ground material/pattern; curb and drainage; fixed traffic furniture (cones, bollards, signs); presence/absence of road markings; building facades with left/right position; distinctive utilities/structures; vegetation layout.

        # Constraints: no people or vehicles; no motion/time/weather; no verbs; no labels or colons; avoid generic words like “pavement/vegetation/building”—prefer specific materials, colors, patterns, counts, and left/right relations.

        # Format: lowercase; semicolon-separated; single line; ≤77 tokens; no extra text.

        # ASSISTANT:"""



        # prompt_template = """Provide an objective, concise description (max 60 tokens) of only the permanent static elements in this image without first-person or subjective language. Focus on:
        # - Building facades, materials, shapes, and relative positions
        # - Road layout (curbs, lanes, intersections, roundabouts)
        # - Static signage, logos, and fixtures
        # - Landscaping elements (hedges, trees, sidewalks)"""




        def caption(image_path: Path) -> str:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(
                text=prompt_template, 
                images=image, 
                return_tensors="pt"
            ).to(model.device) # 优化1: 使用 model.device 保证设备一致性

            # 优化2: 获取模型的结束符 ID
            eos_token_id = processor.tokenizer.eos_token_id
            
            # 优化3: 增加 token 数量并使用更智能的生成参数
            out = model.generate(
                **inputs, 
                max_new_tokens=250,      # 增加到 250，给予充分的生成空间  clip 77个就截断
                do_sample=False,         # 使用贪心解码，保证输出的稳定性
                eos_token_id=eos_token_id  # 告诉模型在生成结束符时自动停止
            )

            full_text = processor.decode(out[0], skip_special_tokens=True)
            # 清理输出，只保留助手的回答
            return full_text.split("ASSISTANT:")[-1].strip()
        # --- 替换到这里结束 ---
        return caption
    
 



    # elif model_name == 'qwen-vl':
    #     prompt_text = """You are a navigation assistant observing this image.
    #     Describe the scene as if you're standing at the camera’s viewpoint.
    #     Use first-person perspective (“I’m at...”, “On my left...”, etc.).
    #     Focus on:
    #     - Unique or permanent landmarks (e.g., logos, building names, statues)
    #     - Directional relationships (e.g., “on my right”, “ahead”, “behind me”)
    #     - Road structure or intersections
    #     - Building materials or signs that help visual recognition

    #     Keep the description concise and natural. Avoid speculation or subjective opinions."""

    # try:
    #     # 优先：Qwen2-VL
    #     from transformers import AutoProcessor, BitsAndBytesConfig
    #     from transformers import Qwen2VLForConditionalGeneration

    #     model_id = "Qwen/Qwen2-VL-7B-Instruct"   # 或 "Qwen/Qwen2-VL-2B-Instruct"
    #     bnb = BitsAndBytesConfig(
    #         load_in_4bit=True, bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    #     )
    #     processor = AutoProcessor.from_pretrained(model_id)
    #     model = Qwen2VLForConditionalGeneration.from_pretrained(
    #         model_id, quantization_config=bnb, torch_dtype=torch.float16, device_map="auto"
    #     )

    #     def caption(image_path: Path) -> str:
    #         image = Image.open(image_path).convert("RGB")
    #         messages = [{"role": "user", "content": [
    #             {"type": "text", "text": prompt_text},
    #             {"type": "image"}
    #         ]}]
    #         text = processor.apply_chat_template(messages, add_generation_prompt=True)
    #         inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    #         with torch.no_grad():
    #             out = model.generate(
    #                 **inputs, max_new_tokens=120, do_sample=False,
    #                 pad_token_id=processor.tokenizer.eos_token_id,
    #                 eos_token_id=processor.tokenizer.eos_token_id
    #             )
    #         gen = out[0, inputs["input_ids"].shape[-1]:]
    #         return processor.decode(gen, skip_special_tokens=True).strip()
    #     return caption

    # except Exception:
    #     # 回退：老 Qwen-VL-Chat
    #     from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    #     model_id = "Qwen/Qwen-VL-Chat"
    #     tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True
    #     ).to(device)

    #     gen_cfg = GenerationConfig(max_new_tokens=120, do_sample=False)
    #     def caption(image_path: Path) -> str:
    #         query = tokenizer.from_list_format([
    #             {'image': str(image_path)},
    #             {'text': prompt_text},
    #         ])
    #         resp, _ = model.chat(tokenizer, query=query, history=None, generation_config=gen_cfg)
    #         return resp.strip().replace("\n", " ")
    #     return caption



    # # --- 从这里开始修改 qwen-vl 部分 ---
    # elif model_name == 'qwen-vl':
    #     # model_id = "Qwen/Qwen-VL-Chat"
    #     model_id = "Qwen/Qwen2-VL-7B-Instruct"
    #     tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_id,
    #         torch_dtype=torch.float16,
    #         low_cpu_mem_usage=True,
    #         trust_remote_code=True
    #     ).to(device)



    #     # # # # 1. 定义与 LLaVA 相同的详细指令  ## 372/493 (20)  216/493 (10)
    #     # prompt_text = """You are a navigation assistant observing this image.
    #     # Describe the scene as if you're standing at the camera’s viewpoint.
    #     # Use first-person perspective (“I’m at...”, “On my left...”, etc.).
    #     # Focus on:
    #     # - Unique or permanent landmarks (e.g., logos, building names, statues)
    #     # - Directional relationships (e.g., “on my right”, “ahead”, “behind me”)
    #     # - Road structure or intersections
    #     # - Building materials or signs that help visual recognition

    #     # Keep the description concise and natural. Avoid speculation or subjective opinions."""


    #     # 235/493 (10)
    #     prompt_text = """You are a navigation assistant observing this image.
    #     Describe the scene as if you're standing at the camera’s viewpoint.
    #     Do **not** mention people, cars, or other sentimental description (weather, feeling).
    #     Use first-person perspective (“I’m at...”, “On my left...”, etc.).
    #     Focus on:
    #     - Unique or permanent landmarks (e.g., logos, building, Pavement & Road Markings, Intersection Geometry)
    #     - Traffic Signs and Signals
    #     - Road structure or intersections
    #     - Vegetation & Street Furniture
    #     - Building materials or signs that help visual recognition

    #     Keep the description concise and natural. Avoid speculation or subjective opinions."""




    #     # prompt_text = """
    #     # You are a navigation assistant standing at the camera’s viewpoint.  
    #     # Describe the scene, focus **only** on permanent, structural elements.  
    #     # Do **not** mention people, cars, or other temporary objects.

    #     # Your description should include:
    #     # - **Pavement & Road Markings**: brick or asphalt surface, crosswalk stripes, lane lines, arrows  
    #     # - **Traffic Signs & Signals**: stop signs, triangular warning signs (red border, white center), traffic lights, directional road signs (e.g. blue sign with arrow)  
    #     # - **Intersection Geometry**: T-junction, four-way intersection, roundabout, curbs, medians  
    #     # - **Vegetation & Street Furniture**: trees, shrubs, benches, planters  
    #     # - **Building Details**: materials (concrete, glass façades, metal panels), visible logos or signage, window patterns  
    #     # - **Other Static Features**: lamp posts, bollards, fire hydrants, drainage grates  

    #     # Keep it concise and factual—just enough detail to uniquely recognize this spot.
    #     # """.strip()



    #     # ## 280/493
    #     # prompt_text = """
    #     # You are a navigation assistant standing at the camera’s viewpoint.  
    #     # Describe the scene and focus **only** on permanent, structural elements.  
    #     # Do **not** mention people, cars, or other temporary objects.

    #     # Your description should include:
    #     # - **Pavement & Road Markings**: brick or asphalt surface, crosswalk stripes, lane lines, arrows  
    #     # - **Traffic Signs & Signals**: stop signs, triangular warning signs (red border, white center), traffic lights, directional road signs (e.g. blue sign with arrow)  
    #     # - **Intersection Geometry**: T-junction, four-way intersection, roundabout, curbs, medians  
    #     # - **Vegetation & Street Furniture**: trees, shrubs, benches, planters  
    #     # - **Building Details**: materials (concrete, glass façades, metal panels), visible logos or signage, window patterns  
    #     # - **Other Static Features**: lamp posts, bollards, fire hydrants, drainage grates  

    #     # Keep it concise and factual—just enough detail to uniquely recognize this spot.
    #     # """.strip()



    #     # ## 280/493
    #     # prompt_text = """
    #     # You are a navigation assistant standing at the camera’s viewpoint.  
    #     # Describe the scene, focusing only on permanent, structural elements.  

    #     # Do NOT mention any pedestrians, cars, or other temporary objects.  
    #     # Focus on:
    #     # - **Landmarks & Logos**: building shapes, company signs, iconic facades, visible logos  
    #     # - **Materials & Textures**: glass reflections, concrete patterns, metal panels, brickwork  
    #     # - **Traffic Infrastructure**: lane markings, crosswalk stripes, traffic lights, road signs  
    #     # - **Intersection & Road Geometry**: curb alignments, medians, turning lanes, roundabouts  
    #     # - **Surface Details**: asphalt cracks, sidewalk edges, painted arrows  

    #     # Keep it concise, factual, and limited to what helps uniquely recognize this location.
    #     # """.strip()


    #     # # 289/493 
    #     # prompt_text = (
    #     #     "<|user|>\n"
    #     #     "You are a location-aware assistant.\n\n"
    #     #     "Please describe only the permanent and static elements in this scene.\n"
    #     #     "Do NOT mention any of the following:\n"
    #     #     "- People or vehicles\n"
    #     #     "- Weather or lighting conditions\n"
    #     #     "- Object states that may change (e.g., open or closed doors, empty parking lots)\n\n"
    #     #     "Focus instead on:\n"
    #     #     "- Buildings and their characteristics (e.g., glass facade, brick walls, signs, logos)\n"
    #     #     "- Fixed landmarks (e.g., fences, poles, statues, staircases)\n"
    #     #     "- Road structure and layout (e.g., intersections, curves, narrow alleys, sidewalks)\n"
    #     #     "- Relative spatial relationships (e.g., on my right, straight ahead)\n\n"
    #     #     "Use a first-person perspective. Write 1–2 concise sentences.\n"
    #     #     "<|assistant|>"
    #     # )



    #     ## 263/493
    #     # prompt_text = """You are a navigation assistant. " 
    #     #     "Describe only the permanent, static elements in the scene. "
    #     #     "Ignore people, vehicles, weather, or temporary states like open/closed gates. "
    #     #     "Use a first-person perspective. "
    #     #     "Output two short sentences describing buildings, landmarks, road layout, or directional relationships."""



    #     def caption(image_path: Path) -> str:
    #         # 2. 在 Qwen-VL 的格式中使用详细指令
    #         query = tokenizer.from_list_format([
    #             {'image': str(image_path)},
    #             {'text': prompt_text}, # 使用新的详细指令
    #         ])
            
    #         # --- 关键修正：获取并修改模型自带的 config ---
    #         # 3. 为 model.chat 添加生成参数，防止描述被截断
    #         # 首先，获取模型自带的 generation_config
    #         generation_config = model.generation_config
    #         # 然后，修改我们需要的参数
    #         generation_config.max_new_tokens = 250  # 增加到 250，给予充分的生成空间 77
    #         generation_config.do_sample = False
            
    #         response, _ = model.chat(
    #             tokenizer, 
    #             query=query, 
    #             history=None,
    #             generation_config=generation_config # 传入修改后的配置对象
    #         )
    #         # --- 修正结束 ---
    #         return response.strip().replace("\n", " ")
    #     return caption
    # # --- 修改到这里结束 ---

    # else:
    #     raise ValueError(f"Unsupported model: {model_name}")














# # Corrected code for generate_captions.py
#     elif model_name == 'qwen-vl':
#         try:
#             # Use the modern Qwen2-VL classes
#             from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

#             model_id = "Qwen/Qwen2-VL-7B-Instruct"
        
#             # Use AutoProcessor which combines the tokenizer and image processor
#             processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            
#             # Use the specific model class for Qwen2-VL
#             model = Qwen2VLForConditionalGeneration.from_pretrained(
#                 model_id,
#                 torch_dtype=torch.float16,
#                 low_cpu_mem_usage=True,
#                 device_map="auto", # Let transformers handle device placement
#                 trust_remote_code=True
#             )


#             prompt_text = """You are a navigation assistant observing this image.
#                         Describe the scene as if you're standing at the camera’s viewpoint.
#                         Do **not** mention people, person, cars, or other sentimental description (Sky, clouds, weather, feeling).
#                         Use first-person perspective (“I’m at...”, “On my left...”, etc.).
#                         Focus on:
#                         - Unique or permanent landmarks (e.g., logos, building, Pavement & Road Markings, Intersection Geometry)
#                         - Traffic Signs and Signals
#                         - Road structure or intersections
#                         - Vegetation & Street Furniture
#                         - Building materials or signs that help visual recognition

#                         Keep the description concise and natural. Avoid speculation or subjective opinions.
#                         **IMPORTANT: The entire description must be a single, complete sentence and strictly under 77 words.**"""
       

#             # # --- 优化后的 Prompt --- 365/493
#             # prompt_text = """You are a navigation assistant observing this image from a first-person viewpoint.

#             # Your task is to describe only the permanent, static elements of the scene.

#             # **Do NOT mention:**
#             # - People, person, cars, or any other vehicles.
#             # - Sky, clouds, weather, or lighting conditions.
#             # - Temporary states like "parked", "busy", "empty", or subjective feelings.

#             # **Focus on:**
#             # - **Landmarks & Buildings:** Unique shapes, materials (brick, glass, shape, structure), signs, or logos.
#             # - **Road & Ground:** Pavement type, lane markings, crosswalks, curbs, and intersection layout.
#             # - **Street Furniture:** Lamp posts, benches, fire hydrants, bollards.
#             # - **Spatial Relationships:** Use terms like "on my left," "ahead of me," "across the street."

#             # Keep the description concise and natural. Avoid speculation or subjective opinions.
#             # **IMPORTANT: The entire description must be a single, complete sentence and strictly under 77 words.**"""
       



#             def caption(image_path: Path) -> str:
#                 image = Image.open(image_path).convert("RGB")
                
#                 # Prepare the prompt and image using the correct chat template
#                 messages = [{"role": "user", "content": [
#                     {"type": "text", "text": prompt_text},
#                     {"type": "image"}
#                 ]}]
                
#                 # --- 从这里开始修改 ---
#                 # 1. 首先，使用 apply_chat_template 生成格式化的文本 prompt
#                 text_prompt = processor.apply_chat_template(
#                     messages,
#                     add_generation_prompt=True,
#                     tokenize=False  # 关键：告诉它返回字符串，而不是token IDs
#                 )

#                 # 2. 然后，将格式化后的文本和图片一起传递给 processor，以创建最终的输入字典
#                 inputs = processor(
#                     text=[text_prompt],  # 注意这里要传入一个列表
#                     images=[image],
#                     return_tensors="pt"
#                 ).to(model.device) # 现在可以安全地对整个字典调用 .to()
#                 # --- 修改结束 ---


#                 # Generate the description
#                 with torch.no_grad():
#                     out = model.generate(
#                         **inputs, 
#                         max_new_tokens=120, 
#                         do_sample=False,
#                         pad_token_id=processor.tokenizer.eos_token_id
#                     )
                
#                 # Decode only the newly generated tokens, skipping the prompt
#                 response_ids = out[0, inputs["input_ids"].shape[-1]:]
#                 return processor.decode(response_ids, skip_special_tokens=True).strip()
            
#             return caption



    elif model_name == 'qwen-vl':
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            model_id = "Qwen/Qwen3-VL-8B-Instruct"
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )



            # prompt_text = """You are a navigation assistant observing this image.
            # Describe the scene as if you're standing at the camera’s viewpoint.
            # Do **not** mention people, person, cars, bus, or other sentimental description (Sky, clouds, weather, feeling).
            # Describe only permanent/static elements, Use first-person perspective (“I’m at...”, “On my left...”, etc.).
            # Focus on:
            # - Unique or permanent landmarks (e.g., logos, building, Pavement & Road Markings, Intersection Geometry)
            # - Traffic Signs and Signals
            # - Road structure or intersections
            # - Vegetation & Street Furniture
            # - Building materials/colors or signs that help visual recognition

            # Keep the description concise and natural. Avoid speculation or subjective opinions.**"""




            # prompt_text = """You are a navigation assistant observing this image.
            #             Describe the scene as if you're standing at the camera’s viewpoint.
            #             Do **not** mention people, person, cars, bus, or other sentimental description (Sky, clouds, weather, feeling).
            #             Describe only permanent/static elements, Use first-person perspective (“I’m at...”, “On my left...”, etc.).
            #             Focus on:
            #            - Unique or permanent landmarks (e.g., logos, building, Pavement & Road Markings, Intersection Geometry)
            #            - Traffic Signs and Signals
            #            - Road structure or intersections
            #            - Vegetation & Street Furniture
            #           - Building materials or signs that help visual recognition

            #            Keep the description concise and natural. Avoid speculation or subjective opinions.
            #            **IMPORTANT: The entire description must be a single, complete sentence and strictly under 77 words.**"""
       


            prompt_text = """
            Task: Act as a Visual Positioning System. Analyze the static environment from a first-person perspective.

            **Strict Constraints (Read Carefully):**
            1. **Total Blindness to Dynamic Objects:** You must completely IGNORE all vehicles (cars, buses, trucks), pedestrians, animals, and movable trash cans. Do not acknowledge their existence.
            2. **Total Blindness to Atmosphere:** Exclude all mentions of sky, clouds, weather (sunny, rain), lighting conditions, or emotional vibe.
            3. **Target Output:** Focus purely on permanent infrastructure.The entire description must be a single, complete sentence and strictly under 77 words.

            **Focus ONLY on:**
            - **Text:** Legible shop names, road signs, street numbers.
            - **Road:** Lane markings (solid/dashed lines), crosswalks, asphalt texture, curbs, manhole covers.
            - **Verticals:** Street lamps, traffic lights, utility poles, bollards.
            - **Buildings:** Facade materials (brick, glass), architectural style, fence types.

            Output format:
            "I am facing [Road feature], with [Building/Landmark] on my left and [Signage/Infrastructure] on my right."
            """


            # # --- 新增：专门用于生成难负样本的 Prompt ---
            # hard_negative_prompt_template = """
            # You are a data augmenter. Your task is to create a "hard negative" description based on an original description and an image.

            # <image>

            # Original (Positive) Description:
            # "{positive_description}"

            # Your task:
            # Rewrite the original description to create a new, subtly incorrect "hard negative" version.
            # Follow these rules STRICTLY:
            # 1.  **Flip Directions**: Change "left" to "right", "right" to "left", "ahead" to "behind".
            # 2.  **Change Colors**: If a color is mentioned (e.g., "red sign"), change it to another plausible color (e.g., "blue sign").
            # 3.  **Swap Landmarks**: If two distinct landmarks are mentioned, swap their positions.
            # 4.  **Keep Structure**: The new description must have the same sentence structure and length as the original.
            # 5.  **Be Subtle**: Make only ONE or TWO changes. The goal is to be confusingly similar.

            # Output only the new, rewritten "hard negative" description.
            # """.strip()

            
           

            # prompt_text = """
            # You are a vision tagger for geo-localization. <image>

            # Return EXACTLY TWO LINES:

            # POS: 5–8 static, discriminative NOUN PHRASES (semicolon-separated, lowercase ascii, <=77 tokens total).
            # Focus on permanent cues only: building, glass facade, entrance, logo/sign text, crosswalk, lane marking, curb, sidewalk, median, roundabout, intersection, staircase, canopy, railing, fence, barrier, bollard, streetlight, utility pole, fire hydrant, manhole, drainage grate, billboard, tree, hedge, planter. 
            # No Chinese; no verbs; no sentences; no dynamic/temporary objects (cars, buses, people, cranes, cones, weather, lighting, time). 
            # Use optional direction tags when clear: (on my left)|(on my right)|(ahead)|(behind)|(overhead)|(near the entrance).

            # HARD_NEG: copy POS items but FLIP ONLY the directions:
            # - (on my left) <-> (on my right)
            # - (ahead) <-> (behind)
            # - (near the entrance) <-> (across the street)
            # Phrases without directions stay unchanged. Keep same count/order and format.

            # Example
            # POS: crosswalk; double yellow lane marking; glass facade (on my right); traffic sign "south entry"; utility pole; fire hydrant (near curb)
            # HARD_NEG: crosswalk; double yellow lane marking; glass facade (on my left); traffic sign "south entry"; utility pole; fire hydrant (across the street)

            # Now output the two lines for this image.
            # """.strip()

            # def _postproc_two_lines(text: str) -> tuple[str, str]:
            #     # 取可见行
            #     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            #     joined = " ".join(lines)
            #     # 允许模型把两行粘在一行的情况：尝试按关键词切割
            #     if "HARD_NEG:" in joined and "POS:" in joined:
            #         # 规范化空格
            #         joined = joined.replace("  ", " ")
            #         pos = joined.split("HARD_NEG:")[0]
            #         hard = joined.split("HARD_NEG:", 1)[1]
            #         pos = pos.split("POS:", 1)[1].strip()
            #         hard = hard.strip()
            #     else:
            #         # 正常两行
            #         if len(lines) >= 2:
            #             pos = lines[0]
            #             hard = lines[1]
            #         elif len(lines) == 1:
            #             pos, hard = lines[0], ""
            #         else:
            #             return "", ""

            #     # 去掉前缀
            #     if pos.lower().startswith("pos:"): pos = pos[4:].strip()
            #     if hard.lower().startswith("hard_neg:"): hard = hard[9:].strip()

            #     # 只保留 ASCII、小写、分号与常见符号
            #     import re, string
            #     allowed = set(string.ascii_lowercase + string.digits + ' ";()-:[]')
            #     def clean(s: str) -> str:
            #         s = s.lower()
            #         s = "".join(ch if ch in allowed or ch==';' else ' ' for ch in s)
            #         # 统一分号分隔，移除多余空格和重复分号
            #         items = [it.strip() for it in s.split(';') if it.strip()]
            #         return "; ".join(items)
            #     return clean(pos), clean(hard)

            # def caption(image_path: Path, positive_description: str) -> str:
            def caption(image_path: Path) -> str:
                image = Image.open(image_path).convert("RGB")
                
                # 将正样本描述填入 prompt 模板
                # prompt_text = hard_negative_prompt_template.format(positive_description=positive_description)
                

                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
                text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
                
                with torch.inference_mode():
                    out = model.generate(
                        **inputs, max_new_tokens=250, do_sample=False,  #96
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )
                new_ids = out[0, inputs["input_ids"].shape[-1]:]
                resp = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                return resp
            return caption
        
        
            # --- caption 函数修改结束 ---
            # def caption(image_path: Path) -> tuple[str, str]:
            #     image = Image.open(image_path).convert("RGB")
            #     messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            #     text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            #     inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
            #     with torch.inference_mode():
            #         out = model.generate(
            #             **inputs, max_new_tokens=96, do_sample=False,
            #             eos_token_id=processor.tokenizer.eos_token_id,
            #             pad_token_id=processor.tokenizer.eos_token_id,
            #         )
            #     new_ids = out[0, inputs["input_ids"].shape[-1]:]
            #     resp = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            #     return _postproc_two_lines(resp)
            # return caption # <--- 添加这一行

            # def caption(image_path: Path) -> str:
            #     image = Image.open(image_path).convert("RGB")
            #     messages = [{
            #         "role": "user",
            #         "content": [
            #             {"type": "image"},  # 先放图或先放文都可，Qwen3 解析两种顺序
            #             {"type": "text", "text": prompt_text},
            #         ],
            #     }]
            #     # 1) 先得到 chat 模板文本（字符串）
            #     text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            #     # 2) 再把文本与图像一起喂给 processor，得到张量
            #     inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

            #     with torch.inference_mode():
            #         out = model.generate(
            #             **inputs, max_new_tokens=77, do_sample=False,
            #             eos_token_id=processor.tokenizer.eos_token_id,
            #             pad_token_id=processor.tokenizer.eos_token_id,
            #         )
            #     new_ids = out[0, inputs["input_ids"].shape[-1]:]
            #     resp = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            #     return resp
            # return caption

        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL model: {e}. Check dependencies and model ID.")
            # Return a function that does nothing to prevent crashes, but logs an error
            return lambda img_path: ""
    
def main(args):
    """
    为查询目录中的图像生成正/负描述，并支持对两个输出CSV文件进行查漏补缺。
    """
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        caption_func = get_captioner(args.model, device)
    except Exception as e:
        logger.error(f"Failed to load model '{args.model}': {e}")
        return

    # # 1. 获取源文件夹中所有应该被处理的图像
    # all_image_paths = sorted(
    #     list(Path(args.query_dir).rglob("*.jpg")) + 
    #     list(Path(args.query_dir).rglob("*.png"))
    # )
    # if not all_image_paths:
    #     logger.error(f"No JPG or PNG images found in {args.query_dir}.")
    #     return
    # logger.info(f"Found a total of {len(all_image_paths)} images in the source directory.")

    # # 2. 分别读取两个CSV中已完成的图片名称
    # pos_path = Path(args.output_csv)
    # neg_path = Path(args.neg_output_csv)

    # def _read_done(p: Path) -> set:
    #     s = set()
    #     if p.exists() and p.stat().st_size > 0:
    #         with open(p, "r", encoding="utf-8") as f:
    #             reader = csv.reader(f)
    #             try:
    #                 next(reader) # 跳过表头
    #                 for row in reader:
    #                     if row: s.add(row[0])
    #             except StopIteration:
    #                 pass
    #     return s

    # pos_done = _read_done(pos_path)
    # neg_done = _read_done(neg_path)
    
    # # 关键逻辑：使用交集(&)找出在两个文件中都存在的、真正完成的图片
    # truly_done = pos_done & neg_done
    # logger.info(f"Found {len(pos_done)} entries in positive CSV and {len(neg_done)} in negative CSV.")
    # logger.info(f"Found {len(truly_done)} images that are fully completed in both files.")

    # # 3. 计算出剩下需要处理的图像列表（所有图片 - 真正完成的图片）
    # remaining_paths = [
    #     p for p in all_image_paths 
    #     if p.relative_to(args.query_dir).as_posix() not in truly_done
    # ]

    # if not remaining_paths:
    #     logger.info("All images have already been processed completely. Nothing to do.")
    #     return
        
    # logger.info(f"Resuming caption generation for {len(remaining_paths)} images that are not fully complete...")

    # # 4. 以追加模式('a')打开文件
    # pos_mode = 'a' if pos_path.exists() and pos_path.stat().st_size > 0 else 'w'
    # neg_mode = 'a' if neg_path.exists() and neg_path.stat().st_size > 0 else 'w'
    
    # # 定义一个记录失败图片的文件
    # failed_log_path = pos_path.parent / "failed_captions.log"

    # with open(pos_path, pos_mode, newline='', encoding='utf-8') as fpos, \
    #      open(neg_path, neg_mode, newline='', encoding='utf-8') as fneg, \
    #      open(failed_log_path, 'a', encoding='utf-8') as ffail:
        
    #     wpos = csv.writer(fpos, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    #     wneg = csv.writer(fneg, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

    #     if pos_mode == 'w': wpos.writerow(['image_name', 'description'])
    #     if neg_mode == 'w': wneg.writerow(['image_name', 'description'])

    #     for img_path in tqdm(remaining_paths, desc=f"Generating missing captions with {args.model}"):
    #         relative_name = img_path.relative_to(args.query_dir).as_posix()
    #         try:
    #             pos_desc, neg_desc = caption_func(img_path)

    #             # 关键修复：只要有一个描述不为空，就尝试写入
    #             if pos_desc or neg_desc:
    #                 # 只有当图片不在已完成列表时才写入，避免重复
    #                 if relative_name not in pos_done:
    #                     wpos.writerow([relative_name, pos_desc])
    #                     fpos.flush()
                    
    #                 if relative_name not in neg_done:
    #                     wneg.writerow([relative_name, neg_desc])
    #                     fneg.flush()
    #             else:
    #                 logger.warning(f"Generated empty descriptions for {relative_name}.")
    #                 ffail.write(f"{relative_name}\n")
    #                 ffail.flush()

    #         except Exception as e:
    #             logger.error(f"Failed to process {relative_name}: {e}")
    #             ffail.write(f"{relative_name} # ERROR: {e}\n")
    #             ffail.flush()


    # logger.info(f"Starting caption generation for images in {args.query_dir} using '{args.model}'...")
    
    # # image_paths = sorted(list(args.query_dir.glob("*.jpg")))
    
    # # --- 关键修改：同时查找 .jpg 和 .png 文件 ---
    # image_paths = sorted(
    #     list(Path(args.query_dir).rglob("*.jpg")) + 
    #     list(Path(args.query_dir).rglob("*.png"))
    # )
    # # --- 修改结束 ---

    # if not image_paths:
    #     logger.error(f"No JPG images found in {args.query_dir}.")
    #     return

    # # --- 关键修正：确保生成标准的 CSV 文件 ---
    # with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
    #     # 1. 明确指定分隔符为逗号
    #     writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        
    #     # 2. 将列名从 'prompt' 修改为 'description'
    #     # writer.writerow(['filepath', 'title'])  # 'image_name', 'description'
    #     writer.writerow(['image_name', 'description'])  # 'image_name', 'description'

    #     processed_count = 0
    #     for img_path in tqdm(image_paths, desc=f"Generating captions with {args.model}"):
    #         try:
    #             description = caption_func(img_path)
    #             if description:
    #                 relative_name = img_path.relative_to(args.query_dir).as_posix()
    #                 writer.writerow([relative_name, description])
    #                 f.flush()
    #             else:
    #                 logger.warning(f"Generated empty description for {img_path.name}.")
    #         except Exception as e:
    #             logger.error(f"Failed to process {img_path.name}: {e}")
            
    #         processed_count += 1
    #         # 每处理 500 张图片，就强制清理一次 CUDA 缓存
    #         if processed_count % 500 == 0:
    #             logger.info(f"Processed {processed_count} images, clearing CUDA cache...")
    #             torch.cuda.empty_cache()
    #     # --- 修改结束 ---




    # --- 关键修改：实现断点续传 ---
    
    # 1. 获取所有需要处理的图像
    all_image_paths = sorted(
        list(Path(args.query_dir).rglob("*.jpg")) + 
        list(Path(args.query_dir).rglob("*.png"))
    )
    if not all_image_paths:
        logger.error(f"No JPG or PNG images found in {args.query_dir}.")
        return

    # 2. 检查已有的输出，找出已经处理过的图像
    processed_images = set()
    output_csv_path = Path(args.output_csv)
    file_exists = output_csv_path.exists()

    if file_exists:
        logger.info(f"Found existing output file: {output_csv_path}. Checking for completed images...")
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                header = next(reader) # 跳过表头
                for row in reader:
                    if row: # 避免空行
                        processed_images.add(row[0])
            except StopIteration:
                logger.warning("Output file is empty.")
        logger.info(f"Found {len(processed_images)} already processed images.")

    # 3. 计算出剩下需要处理的图像列表
    remaining_paths = []
    for img_path in all_image_paths:
        relative_name = img_path.relative_to(args.query_dir).as_posix()
        if relative_name not in processed_images:
            remaining_paths.append(img_path)

    if not remaining_paths:
        logger.info("All images have already been processed. Nothing to do.")
        return
        
    logger.info(f"Resuming caption generation for {len(remaining_paths)} remaining images...")

    # 4. 以追加模式('a')打开文件，如果文件不存在则创建('w')
    file_mode = 'a' if file_exists else 'w'
    with open(output_csv_path, file_mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

        # 如果是新文件，写入表头
        if not file_exists:
            writer.writerow(['image_name', 'description'])

        processed_count = 0
        for img_path in tqdm(remaining_paths, desc=f"Resuming captions with {args.model}"):
            try:
                description = caption_func(img_path)
                if description:
                    relative_name = img_path.relative_to(args.query_dir).as_posix()
                    writer.writerow([relative_name, description])
                    f.flush() # 实时写入，防止再次崩溃时丢失数据
                else:
                    logger.warning(f"Generated empty description for {img_path.name}.")
            except Exception as e:
                logger.error(f"Failed to process {img_path.name}: {e}")
            
            processed_count += 1
            if processed_count % 500 == 0:
                logger.info(f"Processed {processed_count} more images, clearing CUDA cache...")
                torch.cuda.empty_cache()
    # --- 修改结束 ---


    logger.info(f"Successfully generated prompts and saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image captions using various models.")
    parser.add_argument("--query_dir", type=Path, required=True, help="Directory containing query images.")
    parser.add_argument("--output_csv", type=Path, required=True, help="Path to save the output CSV file.")
    parser.add_argument("--model", type=str, default="blip", choices=['blip', 'llava', 'qwen-vl'],
                        help="Model to use for captioning.")
    parser.add_argument("--device", type=str, help="Device to use, e.g., 'cuda' or 'cpu'. Autodetects if not set.")
    

    # parser.add_argument(
    # "--neg_output_csv", type=Path, required=True,
    # help="Path to save the hard-negative captions CSV."
    # )


    main(parser.parse_args())


# def main(args):
#     """
#     基于已有的正样本CSV，为其生成对应的难负样本CSV，并支持断点续传。
#     """
#     device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
#     try:
#         # caption_func 现在需要两个参数
#         caption_func = get_captioner(args.model, device)
#     except Exception as e:
#         logger.error(f"Failed to load model '{args.model}': {e}")
#         return

#     # 1. 检查输入文件是否存在
#     pos_path = Path(args.input_csv)
#     neg_path = Path(args.output_csv) # 现在 output_csv 是指难负样本的输出
#     if not pos_path.exists():
#         logger.error(f"Input positive CSV not found: {pos_path}")
#         return

#     # 2. 读取所有需要处理的正样本
#     positive_pairs = {}
#     with open(pos_path, 'r', encoding='utf-8') as f:
#         reader = csv.reader(f)
#         try:
#             next(reader) # 跳过表头
#             for row in reader:
#                 if row: positive_pairs[row[0]] = row[1]
#         except StopIteration:
#             logger.error(f"Input CSV {pos_path} is empty.")
#             return
#     logger.info(f"Loaded {len(positive_pairs)} positive pairs from {pos_path}.")

#     # 3. 检查已完成的难负样本
#     done_negatives = set()
#     if neg_path.exists():
#         with open(neg_path, 'r', encoding='utf-8') as f:
#             reader = csv.reader(f)
#             try:
#                 next(reader)
#                 for row in reader:
#                     if row: done_negatives.add(row[0])
#             except StopIteration:
#                 pass
#     logger.info(f"Found {len(done_negatives)} already completed hard negatives in {neg_path}.")

#     # 4. 找出需要处理的任务
#     tasks_to_do = {name: desc for name, desc in positive_pairs.items() if name not in done_negatives}
#     if not tasks_to_do:
#         logger.info("All hard negatives have already been generated. Nothing to do.")
#         return
#     logger.info(f"Resuming generation for {len(tasks_to_do)} hard negatives...")

#     # 5. 以追加模式处理
#     file_mode = 'a' if neg_path.exists() and neg_path.stat().st_size > 0 else 'w'
#     failed_log_path = neg_path.parent / "failed_hard_negatives.log"

#     with open(neg_path, file_mode, newline='', encoding='utf-8') as fneg, \
#          open(failed_log_path, 'a', encoding='utf-8') as ffail:
        
#         writer = csv.writer(fneg, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
#         if file_mode == 'w':
#             writer.writerow(['image_name', 'description'])

#         for relative_name, pos_desc in tqdm(tasks_to_do.items(), desc=f"Generating hard negatives with {args.model}"):
#             img_path = Path(args.query_dir) / relative_name
#             if not img_path.exists():
#                 logger.warning(f"Image file not found, skipping: {img_path}")
#                 continue
            
#             try:
#                 neg_desc = caption_func(img_path, pos_desc)
#                 if neg_desc:
#                     writer.writerow([relative_name, neg_desc])
#                     fneg.flush()
#                 else:
#                     logger.warning(f"Generated empty hard negative for {relative_name}.")
#                     ffail.write(f"{relative_name}\n")
#                     ffail.flush()
#             except Exception as e:
#                 logger.error(f"Failed to process {relative_name}: {e}")
#                 ffail.write(f"{relative_name} # ERROR: {e}\n")
#                 ffail.flush()

#     logger.info(f"Successfully generated hard negatives and saved to {neg_path}")



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate hard-negative captions based on existing positive captions.")
#     parser.add_argument("--query_dir", type=Path, required=True, help="Base directory containing the images.")
#     # parser.add_argument("--input_csv", type=Path, required=True, help="Path to the input CSV with POSITIVE captions.")
#     parser.add_argument("--output_csv", type=Path, required=True, help="Path to save the OUTPUT hard-negative CSV.")
#     parser.add_argument("--model", type=str, default="qwen-vl", choices=['blip', 'llava', 'qwen-vl'],
#                         help="Model to use for captioning.")
#     parser.add_argument("--device", type=str, help="Device to use, e.g., 'cuda' or 'cpu'. Autodetects if not set.")
    
#     main(parser.parse_args())



# # python3 -m hloc.generate_captions     --query_dir datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove     --output_csv datasets/SensorKit/clipdata/clipdata_prepare_sparse03_remove/0_train_qwen2.csv     --model qwen-vl













# import argparse
# import os
# import sys
# from pathlib import Path
# import torch
# from PIL import Image
# import csv
# from tqdm import tqdm

# # --- 关键修复：将所有需要的 transformers 类统一在文件顶部导入 ---
# # 这样无论此脚本如何被调用，所有类都已定义
# try:
#     from transformers import (
#         AutoProcessor,
#         BitsAndBytesConfig,
#         BlipForConditionalGeneration,
#         BlipProcessor,
#         LlavaNextForConditionalGeneration,
#         LlavaNextProcessor,
#         LlavaOnevisionForConditionalGeneration,
#         Qwen2VLForConditionalGeneration,
#     )
# except ImportError:
#     raise ImportError(
#         "Please install transformers and its dependencies: \n"
#         "pip install transformers torch torchvision bitsandbytes accelerate"
#     )

# # --- logger 兼容：包内优先，失败则回退到标准 logging ---
# try:
#     from . import logger
# except ImportError:
#     import logging
#     logger = logging.getLogger(__name__)
#     logging.basicConfig(
#         level=logging.INFO, format="[%(asctime)s - %(levelname)s] - %(message)s"
#     )


# def get_captioner(model_name: str, device: str):
#     """
#     根据模型名称加载并返回一个 captioning 函数。
#     """
#     logger.info(f"Loading model '{model_name}' onto device '{device}'...")

#     if model_name == 'blip':
#         # 大模型版更稳（精度更高）
#         processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
#         model = BlipForConditionalGeneration.from_pretrained(
#             "Salesforce/blip-image-captioning-large",
#             torch_dtype=torch.float16
#         ).to(device).eval()

#         def caption(image_path: Path) -> str:
#             image = Image.open(image_path).convert("RGB")
#             inputs = processor(image, return_tensors="pt").to(device)
#             out = model.generate(**inputs, max_new_tokens=75)
#             return processor.decode(out[0], skip_special_tokens=True).strip()
#         return caption

#     elif model_name == 'llava':
#         # LLaVA v1.6 Mistral 7B
#         model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float16
#         )
#         processor = LlavaNextProcessor.from_pretrained(model_id)
#         model = LlavaNextForConditionalGeneration.from_pretrained(
#             model_id,
#             quantization_config=bnb_config,
#             device_map="auto",
#             torch_dtype=torch.float16
#         ).eval()

#         prompt_template = """USER: <image>
#         You are a navigation assistant observing this image.
#         Describe the scene as if you're standing at the camera’s viewpoint.
#         Use first-person perspective (“I’m at...”, “On my left...”, etc.).
#         Focus on:
#         - Unique or permanent landmarks (e.g., logos, building names, statues)
#         - Directional relationships (e.g., “on my right”, “ahead”, “behind me”)
#         - Road structure or intersections
#         - Building materials or signs that help visual recognition

#         Keep the description concise and natural. Avoid speculation or subjective opinions.
#         **IMPORTANT: The entire description must be a single, complete sentence and strictly under 77 words.**
#         ASSISTANT:"""

#         def caption(image_path: Path) -> str:
#             image = Image.open(image_path).convert("RGB")
#             inputs = processor(text=prompt_template, images=image, return_tensors="pt").to(model.device)
#             eos = processor.tokenizer.eos_token_id
#             out = model.generate(**inputs, max_new_tokens=77, do_sample=False, eos_token_id=eos)
#             text = processor.decode(out[0], skip_special_tokens=True)
#             # 尽量只取 ASSISTANT 段
#             if "ASSISTANT:" in text:
#                 text = text.split("ASSISTANT:")[-1].strip()
#             return text
#         return caption

#     elif model_name == 'llava-ov':
#         # LLaVA-OneVision 7B（Qwen2 视觉-语言骨干）
#         model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" # 确保这是正确的 ID
#         processor = AutoProcessor.from_pretrained(model_id)
#         model = LlavaOnevisionForConditionalGeneration.from_pretrained(
#             model_id, torch_dtype=torch.float16, device_map="auto"
#         ).eval()

#         # --- 关键修复：将 prompt 简化为纯粹的指令文本 ---
#         prompt_text = """You are a navigation assistant observing this image.
#         Describe the scene as if you're standing at the camera’s viewpoint.
#         Use first-person perspective (“I’m at...”, “On my left...”, etc.).
#         Focus on:
#         - Unique or permanent landmarks (e.g., logos, building names, statues)
#         - Directional relationships (e.g., “on my right”, “ahead”, “behind me”)
#         - Road structure or intersections
#         - Building materials or signs that help visual recognition

#         Keep the description concise and natural. Avoid speculation or subjective opinions.
#         **IMPORTANT: The entire description must be a single, complete sentence and strictly under 77 words.**"""
#         # --- 修复结束 ---
        


#         def caption(image_path: Path) -> str:
#             image = Image.open(image_path).convert("RGB")
#             # OneVision 使用 chat 格式
#             messages = [{"role": "user", "content": [
#                 {"type": "text", "text": prompt_text},
#                 {"type": "image"}
#             ]}]
#             text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#             inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(model.device)
#             out = model.generate(**inputs, max_new_tokens=77, do_sample=False,
#                                  pad_token_id=processor.tokenizer.eos_token_id)
#             gen = out[0, inputs["input_ids"].shape[-1]:]
#             return processor.decode(gen, skip_special_tokens=True).strip()
#         return caption

#     elif model_name in ('qwen2-vl'):
#         # Qwen2-VL-7B-Instruct
#         # from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
#         model_id = "Qwen/Qwen2-VL-7B-Instruct"
#         processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
#         model = Qwen2VLForConditionalGeneration.from_pretrained(
#             model_id,
#             torch_dtype=torch.float16,
#             low_cpu_mem_usage=True,
#             device_map="auto",
#             trust_remote_code=True
#         ).eval()

#         # prompt_text = (
#         #     "Describe only permanent, static elements in this street-view image. "
#         #     "Do NOT mention people, vehicles, weather/sky/lighting or any temporary state. "
#         #     "Focus on: building facades/materials/shapes/relative positions; road layout; static signage/logos; "
#         #     "curbs/sidewalks/hedges/trees. Use neutral, concise, factual language."
#         # )

#         # prompt_text = """You are a navigation assistant observing this image.
#         #     Describe the scene as if you're standing at the camera’s viewpoint.
#         #     Do **not** mention people, cars, or other sentimental description (weather, feeling).
#         #     Use first-person perspective (“I’m at...”, “On my left...”, etc.).
#         #     Focus on:
#         #     - Unique or permanent landmarks (e.g., logos, building, Pavement & Road Markings, Intersection Geometry)
#         #     - Traffic Signs and Signals
#         #     - Road structure or intersections
#         #     - Vegetation & Street Furniture
#         #     - Building materials or signs that help visual recognition

#         #     Keep the description concise and natural. Avoid speculation or subjective opinions.
#         #     **IMPORTANT: The entire description must be a single, complete sentence and strictly under 77 words.**"""
#         # # --- 修改结束 ---

#         # # 这是一个经过优化的、`通用的静态场景描述 Prompt
#         # prompt_text = """You are a navigation assistant observing this image from a first-person viewpoint.

#         # Your task is to describe only the permanent, static elements of the scene.

#         # **Do NOT mention:**
#         # - People, cars, or any other vehicles.
#         # - Sky, clouds, weather, or lighting conditions.
#         # - Temporary states like "parked", "busy", "empty", or subjective feelings.

#         # **Focus on:**
#         # - **Landmarks & Buildings:** Unique shapes, materials (brick, glass), signs, or logos.
#         # - **Road & Ground:** Pavement type, lane markings, crosswalks, curbs, and intersection layout.
#         # - **Street Furniture:** Lamp posts, benches, fire hydrants, bollards.
#         # - **Spatial Relationships:** Use terms like "on my left," "ahead of me," "across the street."

#         # Keep the description concise and natural. Avoid speculation or subjective opinions.
#         #     **IMPORTANT: The entire description must be a single, complete sentence and strictly under 77 words.**"""
#         # # --- 修改结束 ---`



#         prompt_text = """You are a navigation assistant observing this image from a first-person viewpoint.

#             Your task is to describe only the permanent, static elements of the scene.

#             **Do NOT mention:**
#             - People, cars, or any other vehicles.
#             - Sky, clouds, weather, or lighting conditions.
#             - Temporary states like "parked", "busy", "empty", or subjective feelings.

#             **Focus on:**
#             - **Landmarks & Buildings:** Unique shapes, materials (brick, glass, shape, structure), signs, or logos.
#             - **Road & Ground:** Pavement type, lane markings, crosswalks, curbs, and intersection layout.
#             - **Street Furniture:** Lamp posts, benches, fire hydrants, bollards.
#             - **Spatial Relationships:** Use terms like "on my left," "ahead of me," "across the street."

#             Keep the description concise and natural. Avoid speculation or subjective opinions.
#             **IMPORTANT: The entire description must be a single, complete sentence and strictly under 77 words.**"""
        

#         def caption(image_path: Path) -> str:
#             image = Image.open(image_path).convert("RGB")
#             messages = [{"role": "user", "content": [
#                 {"type": "text", "text": prompt_text},
#                 {"type": "image"}
#             ]}]
#             text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
#             inputs = processor(text=[text_prompt], images=[image], return_tensors="pt").to(model.device)
#             with torch.no_grad():
#                 out = model.generate(
#                     **inputs, max_new_tokens=77, do_sample=False,
#                     pad_token_id=processor.tokenizer.eos_token_id
#                 )
#             gen = out[0, inputs["input_ids"].shape[-1]:]
#             return processor.decode(gen, skip_special_tokens=True).strip()
#         return caption

#     else:
#         raise ValueError(f"Unsupported model: {model_name}")


# def load_image_list(image_list_file: Path):
#     paths = []
#     with open(image_list_file, 'r') as f:
#         for line in f:
#             p = line.strip()
#             if not p:
#                 continue
#             paths.append(Path(p))
#     return paths


# def main(args):
#     """
#     为给定的图像列表或目录生成文本描述，并保存为 CSV 文件。
#     输出列：image_path, description
#     """
#     device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

#     try:
#         caption_func = get_captioner(args.model, device)
#     except Exception as e:
#         logger.error(f"Failed to load model '{args.model}': {e}")
#         return

#     # --- 收集待处理图片 ---
#     image_paths = []
#     if args.image_list:
#         image_paths = load_image_list(args.image_list)
#     elif args.query_dir:
#         image_paths = sorted(list(Path(args.query_dir).rglob("*.jpg")) + list(Path(args.query_dir).rglob("*.png")))
#     else:
#         logger.error("You must provide either --image_list or --query_dir.")
#         return

#     if not image_paths:
#         logger.error("No images found.")
#         return

#     # 统一成绝对路径，后续训练最稳
#     image_paths = [p.resolve() for p in image_paths]

#     logger.info(f"Found {len(image_paths)} images. Starting caption generation with '{args.model}'...")

#     # --- 写 CSV：image_path, description ---
#     out_path = Path(args.output_csv)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     with open(out_path, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
#         writer.writerow(['image_path', 'description'])

#         for img_path in tqdm(image_paths, desc=f"Generating with {args.model}"):
#             try:
#                 if not img_path.exists():
#                     logger.warning(f"Image not found, skip: {img_path}")
#                     continue
#                 desc = caption_func(img_path)
#                 if not desc:
#                     logger.warning(f"Empty caption for {img_path.name}")
#                     continue
#                 writer.writerow([str(img_path), desc])
#             except Exception as e:
#                 logger.error(f"Failed to process {img_path.name}: {e}")

#     logger.info(f"Captions saved to {out_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate image captions using various VLMs.")
#     parser.add_argument("--image_list", type=Path, help="A text file; each line is an image path.")
#     parser.add_argument("--query_dir", type=Path, help="Directory to recursively scan images.")
#     parser.add_argument("--output_csv", type=Path, required=True, help="Path to save the output CSV.")
#     parser.add_argument("--model", type=str,
#                         default="blip",
#                         choices=['blip', 'llava', 'llava-ov', 'qwen2-vl'],
#                         help="VLM model to use for captioning.")
#     parser.add_argument("--device", type=str, help="Device to use, e.g., 'cuda' or 'cpu'.")
#     args = parser.parse_args()
#     main(args)
