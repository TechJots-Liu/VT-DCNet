import os
import json
import base64
import threading
from PIL import Image
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

# ===================== Core configuration =====================
# TODO: Fill in your own API keys. Multiple keys can be listed for failover.
API_KEYS = [
    "YOUR_API_KEY_1",
    # "YOUR_API_KEY_2",
]

# TODO: Fill in the API endpoint of your VLM service.
BASE_URL = "YOUR_BASE_URL"

# TODO: Fill in the folder containing input images.
IMAGE_FOLDER = "path/to/image_folder"

# TODO: Fill in the txt file listing image file names to be processed.
TXT_FILE_PATH = "path/to/image_list.txt"

# TODO: Fill in the output json path for saving generated descriptions.
SAVE_JSON_PATH = "path/to/output.json"

# TODO: Fill in the VLM model name you use.
MODEL_NAME = "YOUR_MODEL_NAME"

BATCH_SIZE = 3
MAX_WORKERS = 3
# Resize the longer side of input images to this value.
MAX_PIXEL = 640
# ==============================================================

lock = threading.Lock()
result_cache = []
current_key_index = 0
all_api_unavailable = False

def init_json():
    """Create an empty json file if it does not exist."""
    if not os.path.exists(SAVE_JSON_PATH):
        with open(SAVE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_processed_images():
    """Load image names that have already been processed (for resume)."""
    processed = set()
    if os.path.exists(SAVE_JSON_PATH):
        with open(SAVE_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for item in data:
                    processed.add(item["image_name"])
            except Exception:
                pass
    return processed


def get_next_client():
    """Return an OpenAI-compatible client using the next API key."""
    global current_key_index
    if current_key_index >= len(API_KEYS):
        current_key_index = 0
    client = OpenAI(api_key=API_KEYS[current_key_index], base_url=BASE_URL)
    current_key_index += 1
    return client


def batch_write_to_json():
    """Write cached results to the output json in batches."""
    global result_cache
    if not result_cache:
        return

    with lock:
        with open(SAVE_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.extend(result_cache)
        with open(SAVE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Batch written: {len(result_cache)} results")
        result_cache = []


def image_to_base64(image_path):
    """Read an image, resize it if needed, and encode it to base64."""
    with Image.open(image_path) as img:
        w, h = img.size
        if max(w, h) > MAX_PIXEL:
            ratio = MAX_PIXEL / max(w, h)
            img = img.resize(
                (int(w * ratio), int(h * ratio)),
                Image.Resampling.LANCZOS
            )
        image_format = img.format.lower()
        buffer = BytesIO()
        img.save(buffer, format=image_format, quality=100)
    return f"data:image/{image_format};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def analyze_image(image_name, processed_images):
    """Generate a structured text description for a single image."""
    global all_api_unavailable

    if all_api_unavailable:
        return None

    if image_name in processed_images:
        print(f"Already processed, skipped: {image_name}")
        return None

    image_path = os.path.join(IMAGE_FOLDER, image_name)
    result = {"image_name": image_name, "description": ""}

    if not os.path.exists(image_path):
        result["description"] = "Failed: image not found"
        print(f"Failed: {image_name} | image not found")
        return result

    for attempt in range(len(API_KEYS)):
        if all_api_unavailable:
            return None

        client = get_next_client()
        try:
            print(f"Processing: {image_name}")
            img_base64 = image_to_base64(image_path)
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "你是由阿里云开发的通义千问视觉语言模型，擅长精准的图像分析与识别，请严格按照用户要求输出结果，不添加额外内容。"
                    },
                    {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_base64}},
                        {"type": "text","text":
                         """
Please carefully examine each pine tree in the image. 
Describe whether there are any abnormal colors of needles (chlorosis, yellowing, reddish-brown). 
If no abnormalities are found, please output 'No infected areas detected.' 
If there is a disease, please describe it in the format as shown below.
Size classification (proportion of whole image):
- Larger: >5%
- Large: 3% - 5%
- Medium: 1% - 3%
- Small: ≤1%  ← Pay special attention to these (isolated discolored crowns)
Location zones: left, upper left, lower left, middle, upper middle, lower middle, right, upper right, lower right
Output format (exactly as below, no extra text):
Example: There are 3 large infected areas in the upper left, upper right, and lower middle parts of the image (each area accounting for about 2.5%); there is 1 medium infected area in the upper middle part of the image (accounting for about 1.5%).

"""
},
                    ]
                }],
                temperature=0.1,
                max_tokens=512,
                stream=False
            )
            result["description"] = completion.choices[0].message.content
            return result

        except Exception as e:
            if attempt == len(API_KEYS) - 1:
                all_api_unavailable = True
                print("All API keys are unavailable. The program stops.")
                result["description"] = f"All API keys unavailable: {str(e)}"
                return result

def main():
    init_json()
    processed_images = load_processed_images()
    print(f"Already processed: {len(processed_images)} images")

    with open(TXT_FILE_PATH, "r", encoding="utf-8") as f:
        image_list = [line.strip() for line in f if line.strip()]

    unprocessed_list = [img for img in image_list if img not in processed_images]
    if not unprocessed_list:
        print("All images have been processed.")
        return
    print(f"To be processed in this run: {len(unprocessed_list)} images")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(analyze_image, name, processed_images)
            for name in unprocessed_list
        ]

        for future in as_completed(futures):
            if all_api_unavailable:
                print("Stopped due to unavailable API keys. Saving results...")
                break

            res = future.result()
            if res:
                result_cache.append(res)
                if len(result_cache) >= BATCH_SIZE:
                    batch_write_to_json()

    if result_cache:
        batch_write_to_json()

    if all_api_unavailable:
        print("Program stopped: no available API key.")
    else:
        print("All processing completed.")


if __name__ == "__main__":
    main()
