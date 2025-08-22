import os
import json
import cv2
import numpy as np
from typing import List

# Mapping of class IDs to class names
LABELS_MAPPING = {
    0: "Car",
    1: "Human"
}

def mapping_class(class_name: str) -> int:
    """Map a class name to its corresponding class ID."""
    try:
        return list(LABELS_MAPPING.keys())[list(LABELS_MAPPING.values()).index(class_name)]
    except ValueError:
        raise ValueError(f"Class name '{class_name}' not found in LABELS_MAPPING")

class InputStream:
    """Helper class to read bits from a binary string."""
    def __init__(self, data: str):
        self.data = data
        self.i = 0
    
    def read(self, size: int) -> int:
        """Read specified number of bits and convert to integer."""
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)

def access_bit(data: bytes, num: int) -> int:
    """Extract a single bit from a byte array at the specified position."""
    base = num // 8
    shift = 7 - (num % 8)
    return (data[base] & (1 << shift)) >> shift

def bytes2bit(data: bytes) -> str:
    """Convert byte array to a binary string."""
    return "".join(str(access_bit(data, i)) for i in range(len(data) * 8))

def brush_to_yolo(rle: List[int], height: int, width: int) -> List[float]:
    """Convert RLE brush labels to YOLO-compatible normalized polygon coordinates.
    
    Args:
        rle (List[int]): Run-Length Encoding data from LabelStudio.
        height (int): Image height.
        width (int): Image width.
        
    Returns:
        List[float]: Normalized (x, y) coordinates for YOLO format.
    """
    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    
    out = np.zeros(num, dtype=np.uint8)
    i = 0
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                out[i] = rle_input.read(word_size)
                i += 1
    image = np.reshape(out, [height, width, 4])[:, :, 3]
    
    _, mask = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            for point in cnt:
                x, y = point[0]
                polygon.extend([x / width, y / height])

    return polygon

def polygon_to_yolo(points: List[List[float]]) -> List[float]:
    polygon = []
    for x, y in points:
        polygon.extend([x / 100, y / 100])
    return polygon

def json_to_yolo(input_file: str, output_dir: str) -> None:
    with open(input_file, "r") as f:
        data = json.load(f)
    
    skipped_labels = []
    for task in data:
        image_name = task["data"]["image"].split("/")[-1].split(".")[0]
        polygons = []
        
        for annotation in task.get("annotations", []):
            for item in annotation["result"]:
                height = item["original_height"]
                width = item["original_width"]
                
                if item.get("type") == "brushlabels":
                    polygon = {
                        "points": brush_to_yolo(item["value"]["rle"], height, width),
                        "class": item["value"]["brushlabels"]
                    }
                elif item.get("type") == "polygonlabels":
                    polygon = {
                        "points": polygon_to_yolo(item["value"]["points"]),
                        "class": item["value"]["polygonlabels"]
                    }
                else:
                    skipped_labels.append({
                        "task_id": task.get("id"),
                        "type": item.get("type"),
                        "id": item.get("id")
                    })
                    continue
                polygons.append(polygon)
        
        with open(os.path.join(output_dir, f"{image_name}.txt"), "w") as f:
            for polygon in polygons:
                pts = polygon["points"]
                class_id = mapping_class(polygon["class"][0])
                f.write(f"{class_id} {' '.join(map(str, pts))}\n")
        
        print(f"Converted: {image_name}.txt")
    
    print("Conversion completed.")
    if skipped_labels:
        print(f"Skipped labels: {skipped_labels}")

if __name__ == "__main__":
    input_file = "brush.json"
    output_dir = "labels"
    os.makedirs(output_dir, exist_ok=True)
    json_to_yolo(input_file, output_dir)