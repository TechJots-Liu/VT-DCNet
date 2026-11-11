import os
import ast
import pandas as pd
from typing import List, Tuple, Dict, Optional

# 按比例定义区域坐标（左上部0.4，中间0.2）
REGIONS = {
    "左上部": (0.0, 0.0, 0.4, 0.4),
    "左下部": (0.0, 0.6, 0.4, 1.0),
    "右上部": (0.6, 0.0, 1.0, 0.4),
    "右下部": (0.6, 0.6, 1.0, 1.0),
    "中部靠上": (0.4, 0.0, 0.6, 0.4),
    "中部靠下": (0.4, 0.6, 0.6, 1.0),
    "中部靠左": (0.0, 0.4, 0.4, 0.6),
    "中部靠右": (0.6, 0.4, 1.0, 0.6),
    "中部": (0.4, 0.4, 0.6, 0.6),
}

REGION_ORDER = ["左上部", "左下部", "右上部", "右下部", 
                "中部靠上", "中部靠下", "中部靠左", "中部靠右", "中部"]

def calculate_overlap_area(bbox: Tuple[float, float, float, float], 
                          region: Tuple[float, float, float, float]) -> float:
    """计算边界框与区域的重叠面积"""
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    reg_x1, reg_y1, reg_x2, reg_y2 = region
    
    # 计算重叠区域的坐标
    x_left = max(bbox_x1, reg_x1)
    y_top = max(bbox_y1, reg_y1)
    x_right = min(bbox_x2, reg_x2)
    y_bottom = min(bbox_y2, reg_y2)
    
    # 无重叠的情况
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    return (x_right - x_left) * (y_bottom - y_top)

def determine_region(bbox: Tuple[float, float, float, float]) -> str:
    """确定边界框所属的区域（重叠面积最大的区域）"""
    max_overlap = -1.0
    best_region = "中部"  # 默认区域
    
    for region_name, region_coords in REGIONS.items():
        overlap = calculate_overlap_area(bbox, region_coords)
        if overlap > max_overlap:
            max_overlap = overlap
            best_region = region_name
    
    return best_region

def classify_area(area_percent: float) -> str:
    """根据面积百分比对区域进行分类"""
    if area_percent < 0.6:
        return "小面积"
    elif area_percent <= 1.5:
        return "中等面积"
    else:
        return "大面积"

def read_bbox_from_txt(txt_path: str) -> Tuple[Optional[List[List[float]]], Optional[str]]:
    """从文本文件中解析边界框数据"""
    try:
        if not os.path.exists(txt_path):
            return None, f"文件不存在: {txt_path}"
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # 处理内容格式
        content = content.replace('\n', '').replace(' ', '')
        
        # 单一边界框情况
        if content.startswith('[') and content.endswith(']') and '],[' not in content:
            bbox = ast.literal_eval(content)
            if isinstance(bbox, list) and len(bbox) == 5:
                return [bbox[1:5]], None
        
        # 多个边界框情况
        if '],[' in content:
            content = f"[{content}]"
            
        bbox_obj = ast.literal_eval(content)
        valid_bboxes = []
        
        if isinstance(bbox_obj, list):
            # 处理列表中的多个边界框
            if all(isinstance(item, list) for item in bbox_obj):
                for bbox in bbox_obj:
                    if len(bbox) == 5:
                        valid_bboxes.append(bbox[1:5])
            # 处理单个边界框
            elif len(bbox_obj) == 5:
                valid_bboxes.append(bbox_obj[1:5])
            else:
                raise ValueError(f"检测框元素数量错误，应为5个，实际为{len(bbox_obj)}个")
        else:
            raise ValueError("边界框必须为列表格式")
            
        return valid_bboxes, None
        
    except Exception as e:
        return None, f"解析错误：{str(e)}"

def generate_disease_report(bboxes: List[List[float]]) -> str:
    """生成病害描述报告"""
    # 初始化数据结构
    region_groups: Dict[str, Dict[str, Dict[str, float]]] = {}  # {区域: {面积类型: {count: 数量, total_area: 总面积}}}
    area_groups = {
        "小面积": {"total_count": 0, "total_area": 0.0, "regions": []},
        "中等面积": {"total_count": 0, "total_area": 0.0, "regions": []},
        "大面积": {"total_count": 0, "total_area": 0.0, "regions": []}
    }

    # 统计检测框数据
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        area_percent = (x2 - x1) * (y2 - y1) * 100
        area_percent = round(area_percent, 1)
        region = determine_region((x1, y1, x2, y2))
        area_type = classify_area(area_percent)

        # 更新区域分组
        if region not in region_groups:
            region_groups[region] = {}
        if area_type not in region_groups[region]:
            region_groups[region][area_type] = {"count": 0, "total_area": 0.0}
        
        region_groups[region][area_type]["count"] += 1
        region_groups[region][area_type]["total_area"] += area_percent

        # 更新面积类型分组
        area_groups[area_type]["total_count"] += 1
        area_groups[area_type]["total_area"] += area_percent
        if region not in area_groups[area_type]["regions"]:
            area_groups[area_type]["regions"].append(region)

    report_parts = []
    processed_regions = set()

    # 1. 处理跨区域同面积合并（只处理那些只有一个面积类型的区域）
    for area_type in ["小面积", "中等面积", "大面积"]:
        area_data = area_groups[area_type]
        if len(area_data["regions"]) >= 2 and area_data["total_count"] >= 2:
            # 只考虑那些只有一个面积类型的区域
            single_area_regions = []
            for region in area_data["regions"]:
                # 检查该区域是否只有一个面积类型
                if len(region_groups[region]) == 1 and area_type in region_groups[region]:
                    single_area_regions.append(region)
                    processed_regions.add((region, area_type))
            
            if len(single_area_regions) >= 2:
                # 按预设顺序排序区域
                sorted_regions = [r for r in REGION_ORDER if r in single_area_regions]
                # 计算这些区域的平均面积
                total_count = sum(region_groups[r][area_type]["count"] for r in sorted_regions)
                total_area = sum(region_groups[r][area_type]["total_area"] for r in sorted_regions)
                avg_area = round(total_area / total_count, 1)
                
                region_str = "、".join(sorted_regions)
                count_str = f"{total_count}个"
                report_parts.append(f"{region_str}共有{count_str}{area_type}感染区域（单个区域面积约占{avg_area}%）")

    # 2. 处理所有区域（包括有多个面积类型的区域）
    for region in REGION_ORDER:
        if region not in region_groups:
            continue
        
        # 收集该区域所有面积类型的描述
        area_descriptions = []
        for area_type in ["小面积", "中等面积", "大面积"]:
            if area_type in region_groups[region] and (region, area_type) not in processed_regions:
                region_area_data = region_groups[region][area_type]
                count = region_area_data["count"]
                avg_area = round(region_area_data["total_area"] / count, 1)
                
                if count == 1:
                    area_descriptions.append(f"1个{area_type}感染区域（面积约占{avg_area}%）")
                else:
                    area_descriptions.append(f"{count}个{area_type}感染区域（单个区域面积约占{avg_area}%）")
        
        if area_descriptions:
            # 拼接同区域的多面积描述
            combined_desc = "和".join(area_descriptions)
            report_parts.append(f"{region}有{combined_desc}")

    # 无检测框情况
    if not report_parts:
        return "图像中未检测到感染区域。"
    
    # 仅在第一句添加"图像"
    report_parts[0] = f"图像{report_parts[0]}"
    
    return "；".join(report_parts) + "。"

def validate_directory(path: str, description: str) -> bool:
    """验证目录是否存在"""
    if not os.path.exists(path):
        print(f"错误：{description}目录不存在 - {path}")
        return False
    if not os.path.isdir(path):
        print(f"错误：{description}路径不是一个目录 - {path}")
        return False
    return True

def batch_process_images(image_dir: str, bbox_dir: str, output_excel: str) -> None:
    """批量处理图像和边界框文件，生成Excel报告"""
    # 验证输入目录
    if not validate_directory(image_dir, "图像"):
        return
    if not validate_directory(bbox_dir, "边界框"):
        return
    
    # 创建输出目录（如果需要）
    output_dir = os.path.dirname(output_excel)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 支持的图像扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and
        os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_files:
        print(f"在图像目录中未找到任何图像文件：{image_dir}")
        return

    # 收集结果数据（使用列表而非频繁concat，提高性能）
    results_data = []
    total_files = len(image_files)
    
    print(f"开始处理，共发现{total_files}个图像文件...")
    
    for i, img_file in enumerate(image_files, 1):
        img_name = os.path.splitext(img_file)[0]
        bbox_txt_path = os.path.join(bbox_dir, f"{img_name}.txt")
        # print(f"处理 {i}/{total_files}: {img_file}")

        # 检查边界框文件是否存在
        if not os.path.exists(bbox_txt_path):
            results_data.append({
                '图片名称': img_file,
                '病害描述': f'未找到对应边界框文件：{img_name}.txt'
            })
            continue

        # 读取和解析边界框
        bboxes, err = read_bbox_from_txt(bbox_txt_path)
        if err or not bboxes:
            results_data.append({
                '图片名称': img_file,
                '病害描述': f'边界框解析失败：{err}'
            })
            continue

        # 生成病害描述
        try:
            description = generate_disease_report(bboxes)
            results_data.append({
                '图片名称': img_file,
                '病害描述': description
            })
        except Exception as e:
            results_data.append({
                '图片名称': img_file,
                '病害描述': f'处理图片时出错：{str(e)}'
            })

    # 创建DataFrame并保存结果
    try:
        results = pd.DataFrame(results_data)
        results.to_excel(output_excel, index=False)
        print(f"\n处理完成，共处理{total_files}个文件，结果已保存至：{output_excel}")
    except PermissionError:
        # 权限不足时保存到桌面
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        output_excel = os.path.join(desktop_path, "病害检测结果.xlsx")
        results = pd.DataFrame(results_data)
        results.to_excel(output_excel, index=False)
        print(f"\n权限不足，结果已保存至桌面：{output_excel}")
    except Exception as e:
        print(f"\n保存结果失败：{str(e)}")

if __name__ == "__main__":
    # 配置路径
    image_directory = r"F:\PWD\splitPWD\7011样本处理\640x640\image"
    bbox_directory = r"F:\PWD\splitPWD\7011样本处理\640x640\label"
    output_excel_path = r"F:\PWD\splitPWD\7011样本处理\640x640\病害检测结果.xlsx"
    
    # 执行批量处理
    batch_process_images(image_directory, bbox_directory, output_excel_path)
