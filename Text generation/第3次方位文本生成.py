import os
import json
import base64
import threading
from PIL import Image
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

# ===================== 🔥 核心配置（你只需要修改这里）=====================
API_KEYS = [
    # "sk-1472ac3178f14d3db858914f4951809b",
    # "sk-1f653d0b6651403a8384a8035ddab483",#钟
    # "sk-0b0c838360744aa58ef490f631a73ef7", #黄
    "sk-90fd205853e84ebca71687d3be5ffbe8", #晶
]
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

IMAGE_FOLDER = "/data/lzq/code/twothesis/实验/BLIP-2/ssss"
TXT_FILE_PATH = "第三次描述/two/第sssss.txt"
SAVE_JSON_PATH = "第三次描述/two/第sssss.json"

BATCH_SIZE = 3
MAX_WORKERS = 3
# VL模型最佳尺寸：无人机遥感图640px足够清晰，速度极快
MAX_PIXEL = 640
# ======================================================================

lock = threading.Lock()
result_cache = []
current_key_index = 0
all_api_unavailable = False

def init_json():
    if not os.path.exists(SAVE_JSON_PATH):
        with open(SAVE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

def load_processed_images():
    processed = set()
    if os.path.exists(SAVE_JSON_PATH):
        with open(SAVE_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for item in data:
                    processed.add(item["image_name"])
            except:
                pass
    return processed

def get_next_client():
    global current_key_index
    if current_key_index >= len(API_KEYS):
        current_key_index = 0
    client = OpenAI(api_key=API_KEYS[current_key_index], base_url=BASE_URL)
    current_key_index += 1
    return client

def batch_write_to_json():
    global result_cache
    if not result_cache:
        return

    with lock:
        with open(SAVE_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.extend(result_cache)
        with open(SAVE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 批量写入 {len(result_cache)} 条结果到JSON")
        result_cache = []

def image_to_base64(image_path):
    with Image.open(image_path) as img:
        w, h = img.size
        if max(w, h) > MAX_PIXEL:
            ratio = MAX_PIXEL / max(w, h)
            img = img.resize((int(w*ratio), int(h*ratio)), Image.Resampling.LANCZOS)
        image_format = img.format.lower()
        buffer = BytesIO()
        img.save(buffer, format=image_format, quality=100)
    return f"data:image/{image_format};base64,{base64.b64encode(buffer.getvalue()).decode()}"

# -------------------------- 核心：VL模型专用分析函数 --------------------------
def analyze_image(image_name, processed_images):
    global all_api_unavailable
    
    if all_api_unavailable:
        return None
    
    if image_name in processed_images:
        print(f"⏭️ 已处理，跳过：{image_name}")
        return None

    image_path = os.path.join(IMAGE_FOLDER, image_name)
    result = {"image_name": image_name, "description": ""}

    if not os.path.exists(image_path):
        result["description"] = "处理失败：图片不存在"
        print(f"❌ {image_name} | {result['description']}")
        return result

    for attempt in range(len(API_KEYS)):
        if all_api_unavailable:
            return None
            
        client = get_next_client()
        try:
            print(f"🔄 处理中：{image_name}")
            img_base64 = image_to_base64(image_path)
            completion = client.chat.completions.create(
                # ✅ 固定为你想要的视觉专用模型
                model="qwen3.6-plus",
                messages=[
                    {
                        "role": "system",
                        "content": "你是由阿里云开发的通义千问视觉语言模型，擅长精准的图像分析与识别，请严格按照用户要求输出结果，不添加额外内容。"
                    },
                    {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_base64}},
                        # ✅ VL模型专用提示词（解决重复+精准视觉识别）
                        {"type": "text","text":
                         """
Please carefully examine each pine tree in the image. 
Describe whether there are any abnormal colors of needles (chlorosis, yellowing, reddish-brown), or local wilting of branches, etc., as early signs of pine wood nematode disease. 
If no abnormalities are found, please output 'No infected areas detected.' 
If there is a disease, please describe it in the format as shown below.
Size classification (proportion of whole image):
- Larger: >5%
- Large: 3% - 5%
- Medium: 1% - 3%
- Small: ≤1%  ← Pay special attention to these (isolated discolored crowns)
Location zones: left, upper left, lower left, middle, upper middle, lower middle, right, upper right, lower right
Output format (exactly as below, no extra text):
Example: There are 3 large infected areas in the upper left, upper right, and lower middle parts of the image (each area accounting for about 2.5%); there is 1 medium infected area in the upper middle part of the image (accounting for about 1.5%); there are 2 small infected areas in the middle part of the image (each area accounting for about 0.2%).

"""
# """

# You are an expert in remote sensing forestry health assessment.  Analyze the provided image for pine wood nematode disease.

# Task: Identify all infected areas showing dead pine trees in red, red-brown, yellow, yellow-brown, or gray-brown.

# Step-by-step thinking (enable thinking mode):
# 1.  Scan the entire image for any discolored tree crowns.
# 2.  For each candidate, verify it's not bare soil/shadow/dry grass.
# 3.  If none found, output exactly: "No infected areas detected."

# Size classification (proportion of whole image):
# - Larger: >5%
# - Large: 3% - 5%
# - Medium: 1% - 3%
# - Small: ≤1%  ← Pay special attention to these (isolated discolored crowns)

# Location zones: left, upper left, lower left, middle, upper middle, lower middle, right, upper right, lower right

# Output format (exactly as below, no extra text):
# Example: There are 3 large infected areas in the upper left, upper right, and lower middle parts of the image (each area accounting for about 2.5%); there is 1 medium infected area in the upper middle part of the image (accounting for about 1.5%); there are 2 small infected areas in the middle part of the image (each area accounting for about 0.2%).

# """
},
                    ]
                }],
                # ✅ VL模型最优参数：准+快+不重复
                temperature=0.1,
                max_tokens=512,
                stream=False
            )
            result["description"] = completion.choices[0].message.content
            print(f"✅ {image_name} | 处理完成")
            return result

        except Exception as e:
            if attempt == len(API_KEYS) - 1:
                all_api_unavailable = True
                print(f"\n❌ 严重错误：所有API Key均失效/无额度！程序立即停止！")
                result["description"] = f"所有API Key失效：{str(e)}"
                return result

def main():
    init_json()
    processed_images = load_processed_images()
    print(f"📊 已处理图片：{len(processed_images)} 张")

    with open(TXT_FILE_PATH, "r", encoding="utf-8") as f:
        image_list = [line.strip() for line in f if line.strip()]

    unprocessed_list = [img for img in image_list if img not in processed_images]
    if not unprocessed_list:
        print("🎉 所有图片已处理完成！")
        return
    print(f"🚀 本次待处理：{len(unprocessed_list)} 张")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(analyze_image, name, processed_images) for name in unprocessed_list]
        
        for future in as_completed(futures):
            if all_api_unavailable:
                print("\n🛑 程序已因API失效停止，正在保存剩余数据...")
                break
                
            res = future.result()
            if res:
                result_cache.append(res)
                if len(result_cache) >= BATCH_SIZE:
                    batch_write_to_json()

    if result_cache:
        batch_write_to_json()

    if all_api_unavailable:
        print("\n❌ 程序已停止：无可用API Key")
    else:
        print("\n🎉 本次处理完成！")

if __name__ == "__main__":
    main()