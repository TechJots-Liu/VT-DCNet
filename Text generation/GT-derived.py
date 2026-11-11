import os
import ast
import pandas as pd
from typing import List, Tuple, Dict, Optional

# TODO: Adjust region definitions if your annotation protocol differs.
# Regions are defined as normalized coordinates (x1, y1, x2, y2).
REGIONS = {
    "upper left": (0.0, 0.0, 0.4, 0.4),
    "lower left": (0.0, 0.6, 0.4, 1.0),
    "upper right": (0.6, 0.0, 1.0, 0.4),
    "lower right": (0.6, 0.6, 1.0, 1.0),
    "upper middle": (0.4, 0.0, 0.6, 0.4),
    "lower middle": (0.4, 0.6, 0.6, 1.0),
    "middle left": (0.0, 0.4, 0.4, 0.6),
    "middle right": (0.6, 0.4, 1.0, 0.6),
    "middle": (0.4, 0.4, 0.6, 0.6),
}

REGION_ORDER = [
    "upper left", "lower left", "upper right", "lower right",
    "upper middle", "lower middle", "middle left", "middle right", "middle"
]


def calculate_overlap_area(bbox: Tuple[float, float, float, float],
                           region: Tuple[float, float, float, float]) -> float:
    """Calculate the overlap area between a bounding box and a region."""
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
    """Assign a bounding box to the region with the largest overlap."""
    max_overlap = -1.0
    best_region = "middle"

    for region_name, region_coords in REGIONS.items():
        overlap = calculate_overlap_area(bbox, region_coords)
        if overlap > max_overlap:
            max_overlap = overlap
            best_region = region_name

    return best_region


def classify_area(area_percent: float) -> str:
    """Classify an infected area by its percentage of the whole image."""
    # TODO: Adjust the thresholds so that they match the structured-text template in the manuscript.
    if area_percent < 0.6:
        return "small"
    elif area_percent <= 1.5:
        return "medium"
    else:
        return "large"


def read_bbox_from_txt(txt_path: str) -> Tuple[Optional[List[List[float]]], Optional[str]]:
    """Parse bounding boxes from a txt file.

    The expected format is a list of [class_id, x1, y1, x2, y2],
    where coordinates are normalized to [0, 1].
    """
    try:
        if not os.path.exists(txt_path):
            return None, f"File not found: {txt_path}"

        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        content = content.replace('\n', '').replace(' ', '')

        # Single bounding box
        if content.startswith('[') and content.endswith(']') and '],[' not in content:
            bbox = ast.literal_eval(content)
            if isinstance(bbox, list) and len(bbox) == 5:
                return [bbox[1:5]], None

        # Multiple bounding boxes
        if '],[' in content:
            content = f"[{content}]"

        bbox_obj = ast.literal_eval(content)
        valid_bboxes = []

        if isinstance(bbox_obj, list):
            if all(isinstance(item, list) for item in bbox_obj):
                for bbox in bbox_obj:
                    if len(bbox) == 5:
                        valid_bboxes.append(bbox[1:5])
            elif len(bbox_obj) == 5:
                valid_bboxes.append(bbox_obj[1:5])
            else:
                raise ValueError(
                    f"Each bounding box should contain 5 values, got {len(bbox_obj)}"
                )
        else:
            raise ValueError("Bounding boxes must be provided as a list")

        return valid_bboxes, None

    except Exception as e:
        return None, f"Parsing error: {str(e)}"


def generate_disease_report(bboxes: List[List[float]]) -> str:
    """Generate a structured disease description from bounding boxes."""
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
    processed_regions = set()

    # Merge regions that share the same area type
    for area_type in ["small", "medium", "large"]:
        area_data = area_groups[area_type]
        if len(area_data["regions"]) >= 2 and area_data["total_count"] >= 2:
            single_area_regions = []
            for region in area_data["regions"]:
                if len(region_groups[region]) == 1 and area_type in region_groups[region]:
                    single_area_regions.append(region)
                    processed_regions.add((region, area_type))

            if len(single_area_regions) >= 2:
                sorted_regions = [r for r in REGION_ORDER if r in single_area_regions]
                total_count = sum(region_groups[r][area_type]["count"] for r in sorted_regions)
                total_area = sum(region_groups[r][area_type]["total_area"] for r in sorted_regions)
                avg_area = round(total_area / total_count, 1)

                region_str = ", ".join(sorted_regions)
                report_parts.append(
                    f"There are {total_count} {area_type} infected areas in {region_str} "
                    f"(each accounting for about {avg_area}%)"
                )

    # Handle all remaining regions
    for region in REGION_ORDER:
        if region not in region_groups:
            continue

        area_descriptions = []
        for area_type in ["small", "medium", "large"]:
            if area_type in region_groups[region] and (region, area_type) not in processed_regions:
                region_area_data = region_groups[region][area_type]
                count = region_area_data["count"]
                avg_area = round(region_area_data["total_area"] / count, 1)

                if count == 1:
                    area_descriptions.append(
                        f"1 {area_type} infected area (accounting for about {avg_area}%)"
                    )
                else:
                    area_descriptions.append(
                        f"{count} {area_type} infected areas "
                        f"(each accounting for about {avg_area}%)"
                    )

        if area_descriptions:
            combined_desc = " and ".join(area_descriptions)
            report_parts.append(f"There are {combined_desc} in the {region} part of the image")

    if not report_parts:
        return "No infected areas detected."

    return "; ".join(report_parts) + "."


def validate_directory(path: str, description: str) -> bool:
    """Check whether a directory exists."""
    if not os.path.exists(path):
        print(f"Error: {description} directory does not exist - {path}")
        return False
    if not os.path.isdir(path):
        print(f"Error: {description} path is not a directory - {path}")
        return False
    return True


def batch_process_images(image_dir: str, bbox_dir: str, output_excel: str) -> None:
    """Batch process images and bounding-box files, then save the results as Excel."""
    # TODO: Fill in image_dir and bbox_dir before running.
    if not validate_directory(image_dir, "image"):
        return
    if not validate_directory(bbox_dir, "bounding box"):
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
        print(f"No image files found in {image_dir}")
        return

    results_data = []
    total_files = len(image_files)

    print(f"Start processing {total_files} image files...")

    for i, img_file in enumerate(image_files, 1):
        img_name = os.path.splitext(img_file)[0]
        bbox_txt_path = os.path.join(bbox_dir, f"{img_name}.txt")

        if not os.path.exists(bbox_txt_path):
            results_data.append({
                'image_name': img_file,
                'disease_description': f'Bounding box file not found: {img_name}.txt'
            })
            continue

        bboxes, err = read_bbox_from_txt(bbox_txt_path)
        if err or not bboxes:
            results_data.append({
                'image_name': img_file,
                'disease_description': f'Failed to parse bounding boxes: {err}'
            })
            continue

        try:
            description = generate_disease_report(bboxes)
            results_data.append({
                'image_name': img_file,
                'disease_description': description
            })
        except Exception as e:
            results_data.append({
                'image_name': img_file,
                'disease_description': f'Error while processing image: {str(e)}'
            })

    try:
        results = pd.DataFrame(results_data)
        results.to_excel(output_excel, index=False)
        print(f"Finished processing {total_files} files. Results saved to: {output_excel}")
    except PermissionError:
        # TODO: Adjust the fallback path if needed.
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        output_excel = os.path.join(desktop_path, "disease_detection_results.xlsx")
        results = pd.DataFrame(results_data)
        results.to_excel(output_excel, index=False)
        print(f"Permission denied. Results saved to desktop: {output_excel}")
    except Exception as e:
        print(f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    # TODO: Fill in your own paths.
    image_directory = r"path/to/image_folder"
    bbox_directory = r"path/to/label_folder"
    output_excel_path = r"path/to/output_excel.xlsx"

    batch_process_images(image_directory, bbox_directory, output_excel_path)
