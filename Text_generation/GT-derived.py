import os
import pandas as pd
from typing import List, Tuple, Dict, Optional

# Define regions by normalized proportion
REGIONS = {
    "upper left": (0.0, 0.0, 0.4, 0.4),
    "lower left": (0.0, 0.6, 0.4, 1.0),
    "upper right": (0.6, 0.0, 1.0, 0.4),
    "lower right": (0.6, 0.6, 1.0, 1.0),
    "upper middle": (0.4, 0.0, 0.6, 0.4),
    "lower middle": (0.4, 0.6, 0.6, 1.0),
    "middle left": (0.0, 0.4, 0.4, 0.6),
    "middle right": (0.6, 0.4, 1.0, 0.6),
    "center": (0.4, 0.4, 0.6, 0.6),
}

REGION_ORDER = [
    "upper left", "lower left", "upper right", "lower right",
    "upper middle", "lower middle", "middle left", "middle right", "center"
]

def calculate_overlap_area(bbox: Tuple[float, float, float, float],
                           region: Tuple[float, float, float, float]) -> float:
    """Calculate overlap area between bbox and region"""
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    reg_x1, reg_y1, reg_x2, reg_y2 = region

    x_left = max(bbox_x1, reg_x1)
    y_top = max(bbox_y1, reg_y1)
    x_right = min(bbox_x2, reg_x2)
    y_bottom = min(bbox_y2, reg_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)


def determine_region(bbox: Tuple[float, float, float, float]) -> str:
    """Return region name with maximum overlap"""
    max_overlap = -1.0
    best_region = "center"

    for region_name, region_coords in REGIONS.items():
        overlap = calculate_overlap_area(bbox, region_coords)
        if overlap > max_overlap:
            max_overlap = overlap
            best_region = region_name

    return best_region


def classify_area(area_percent: float) -> str:
    """Classify bbox area level"""
    if area_percent < 0.6:
        return "small"
    elif area_percent <= 1.5:
        return "medium"
    else:
        return "large"


def read_bbox_from_txt(txt_path: str) -> Tuple[Optional[List[List[float]]], Optional[str]]:
    """Parse YOLO format txt label (cls cx cy w h)"""
    try:
        if not os.path.exists(txt_path):
            return None, f"File not found: {txt_path}"

        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        valid_bboxes = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                return None, f"Invalid line (not 5 values): {line}"
            cls, cx, cy, bw, bh = map(float, parts)
            # YOLO (cx, cy, w, h) -> x1 y1 x2 y2
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            valid_bboxes.append([x1, y1, x2, y2])

        return valid_bboxes, None

    except Exception as e:
        return None, f"Parse error: {str(e)}"

def generate_disease_report(bboxes: List[List[float]]) -> str:
    """Generate English disease description report, strictly match your sentence template"""
    region_groups: Dict[str, Dict[str, Dict[str, float]]] = {}
    area_groups = {
        "small": {"total_count": 0, "total_area": 0.0, "regions": []},
        "medium": {"total_count": 0, "total_area": 0.0, "regions": []},
        "large": {"total_count": 0, "total_area": 0.0, "regions": []}
    }

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        area_percent = (x2 - x1) * (y2 - y1) * 100
        area_percent = round(area_percent, 1)
        region = determine_region((x1, y1, x2, y2))
        area_type = classify_area(area_percent)

        if region not in region_groups:
            region_groups[region] = {}
        if area_type not in region_groups[region]:
            region_groups[region][area_type] = {"count": 0, "total_area": 0.0}

        region_groups[region][area_type]["count"] += 1
        region_groups[region][area_type]["total_area"] += area_percent

        area_groups[area_type]["total_count"] += 1
        area_groups[area_type]["total_area"] += area_percent
        if region not in area_groups[area_type]["regions"]:
            area_groups[area_type]["regions"].append(region)

    report_parts = []
    processed = set()

    # Stage1: same area type, multiple regions (merge)
    for area_type in ["small", "medium", "large"]:
        data = area_groups[area_type]
        if len(data["regions"]) >= 2 and data["total_count"] >= 2:
            candidate_regions = []
            for r in data["regions"]:
                if len(region_groups[r]) == 1 and area_type in region_groups[r]:
                    candidate_regions.append(r)
                    processed.add((r, area_type))
            if len(candidate_regions) >= 2:
                sorted_regions = [r for r in REGION_ORDER if r in candidate_regions]
                cnt = sum(region_groups[r][area_type]["count"] for r in sorted_regions)
                sum_area = sum(region_groups[r][area_type]["total_area"] for r in sorted_regions)
                avg = round(sum_area / cnt, 1)
                if len(sorted_regions) >= 3:
                    reg_str = ", ".join(sorted_regions[:-1]) + ", and " + sorted_regions[-1]
                else:
                    reg_str = " and ".join(sorted_regions)
                report_parts.append(
                    f"There are {cnt} {area_type} infected areas in the {reg_str} parts of the image (each area accounting for about {avg}%)"
                )

    # Stage2: per region, each area type is an independent clause (match your style)
    for region in REGION_ORDER:
        if region not in region_groups:
            continue
        for at in ["small", "medium", "large"]:
            if at in region_groups[region] and (region, at) not in processed:
                d = region_groups[region][at]
                c = d["count"]
                avg = round(d["total_area"] / c, 1)
                if c == 1:
                    part = f"There is 1 {at} infected area in the {region} part (accounting for about {avg}%)"
                else:
                    part = f"There are {c} {at} infected areas in the {region} part (each area accounting for about {avg}%)"
                report_parts.append(part)

    if not report_parts:
        return "No targets detected in the image."

    joined = "; ".join(report_parts) + "."
    joined = joined.replace("; There", "; there")
    return joined


def validate_directory(path: str, description: str) -> bool:
    """Check if directory exists"""
    if not os.path.exists(path):
        print(f"Error: {description} directory does not exist - {path}")
        return False
    if not os.path.isdir(path):
        print(f"Error: {description} path is not a directory - {path}")
        return False
    return True


def batch_process_images(image_dir: str, bbox_dir: str, output_excel: str) -> None:
    """Batch process images and labels, export excel"""
    if not validate_directory(image_dir, "Image"):
        return
    if not validate_directory(bbox_dir, "Label"):
        return

    output_dir = os.path.dirname(output_excel)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and
           os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_files:
        print(f"No image files found in: {image_dir}")
        return

    results_data = []
    total_files = len(image_files)
    print(f"Start processing, total {total_files} images...")

    for i, img_file in enumerate(image_files, 1):
        img_name = os.path.splitext(img_file)[0]
        bbox_txt_path = os.path.join(bbox_dir, f"{img_name}.txt")

        if not os.path.exists(bbox_txt_path):
            results_data.append({
                'Image Name': img_file,
                'Disease Description': f'Corresponding label file not found: {img_name}.txt'
            })
            continue

        bboxes, err = read_bbox_from_txt(bbox_txt_path)
        if err or not bboxes:
            results_data.append({
                'Image Name': img_file,
                'Disease Description': f'Label parse failed: {err}'
            })
            continue

        try:
            description = generate_disease_report(bboxes)
            results_data.append({
                'Image Name': img_file,
                'Disease Description': description
            })
        except Exception as e:
            results_data.append({
                'Image Name': img_file,
                'Disease Description': f'Processing error: {str(e)}'
            })

    try:
        results = pd.DataFrame(results_data)
        results.to_excel(output_excel, index=False)
        print(f"\nFinished. Results saved to: {output_excel}")
    except PermissionError:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        output_excel = os.path.join(desktop_path, "Disease_Detection_Result.xlsx")
        results = pd.DataFrame(results_data)
        results.to_excel(output_excel, index=False)
        print(f"\nPermission denied, saved to desktop: {output_excel}")
    except Exception as e:
        print(f"\nSave failed: {str(e)}")


if __name__ == "__main__":
    image_directory = r"..\images"
    bbox_directory = r"..\labels"
    output_excel_path = r"..\Disease_Detection_Result.xlsx"

    batch_process_images(image_directory, bbox_directory, output_excel_path)
