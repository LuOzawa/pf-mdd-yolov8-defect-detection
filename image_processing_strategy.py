"""
PF-MDD Wide-Format Image Processing Strategy
Handling 4096x256 images for YOLOv8 training

Strategies:
1. Tiling Strategy - Split into overlapping tiles (RECOMMENDED)
2. Adaptive Padding - Resize with aspect ratio preservation
3. Multi-Scale Processing - Process at multiple resolutions
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time
import json

from project_config import BASE_DIR, preferred_training_dataset_dir


class WideImageProcessor:
    """
    Specialized processor for wide-format industrial images (4096x256)
    
    Key Challenges:
    - Extreme aspect ratio (16:1)
    - Standard YOLO expects square/near-square inputs
    - Defects can be small and elongated
    
    Solutions:
    1. Tiling: Divide into manageable chunks
    2. Smart padding: Minimize information loss
    3. Multi-scale: Capture defects at different scales
    """
    
    def __init__(self, 
                 original_size: Tuple[int, int] = (4096, 256),
                 target_tile_size: int = 640,
                 overlap_ratio: float = 0.25):
        """
        Initialize processor
        
        Args:
            original_size: Original image dimensions (width, height)
            target_tile_size: Target tile dimension for YOLO
            overlap_ratio: Overlap between adjacent tiles (0-0.5)
        """
        self.original_width, self.original_height = original_size
        self.target_size = target_tile_size
        self.overlap_ratio = overlap_ratio
        
        # Calculate optimal tiling parameters
        self._calculate_tiling_params()
    
    def _calculate_tiling_params(self):
        """Calculate optimal tiling parameters for the image dimensions"""
        
        # Number of tiles needed horizontally
        effective_width = self.original_width * (1 + self.overlap_ratio)
        self.n_tiles_horizontal = max(1, int(np.ceil(effective_width / self.target_size)))
        
        # Adjust step size for even distribution
        if self.n_tiles_horizontal > 1:
            total_overlap = (self.n_tiles_horizontal - 1) * self.target_size * self.overlap_ratio
            available_space = self.original_width - self.target_size
            self.step_x = (available_space - total_overlap) / (self.n_tiles_horizontal - 1) if self.n_tiles_horizontal > 1 else 0
        else:
            self.step_x = 0
        
        print(f"[Tiling Configuration]")
        print(f"  Original size: {self.original_width}x{self.original_height}")
        print(f"  Target tile: {self.target_size}x{self.target_size}")
        print(f"  Overlap ratio: {self.overlap_ratio*100:.0f}%")
        print(f"  Tiles needed: {self.n_tiles_horizontal} (horizontal)")
        print(f"  Step size: {self.step_x:.1f} pixels")
    
    def tile_image_with_overlap(self, 
                                 image: np.ndarray,
                                 return_metadata: bool = True) -> List[Tuple[np.ndarray, Optional[Dict]]]:
        """
        Split wide image into overlapping tiles
        
        Args:
            image: Input image (H, W, C)
            return_metadata: Whether to include position metadata
            
        Returns:
            List of (tile_image, metadata_or_None) tuples
        """
        h, w = image.shape[:2]
        tiles = []
        
        # Calculate actual step based on desired number of tiles
        if self.n_tiles_horizontal == 1:
            # Single tile covering entire width
            x_positions = [0]
        else:
            # Distribute tiles evenly
            x_positions = []
            for i in range(self.n_tiles_horizontal):
                if i == 0:
                    x_pos = 0
                elif i == self.n_tiles_horizontal - 1:
                    x_pos = w - self.target_size
                else:
                    x_pos = int(i * (w - self.target_size) / (self.n_tiles_horizontal - 1))
                x_positions.append(x_pos)
        
        for idx, x_start in enumerate(x_positions):
            # Ensure we don't go out of bounds
            x_end = min(x_start + self.target_size, w)
            
            if x_end - x_start < self.target_size // 2:
                continue
            
            # Extract region
            tile = image[:, x_start:x_end].copy()
            
            # Resize to exact target size if needed
            if tile.shape[1] != self.target_size or tile.shape[0] != self.target_size:
                tile = cv2.resize(tile, (self.target_size, self.target_size), 
                                 interpolation=cv2.INTER_LINEAR)
            
            metadata = None
            if return_metadata:
                actual_scale_x = self.target_size / (x_end - x_start) if x_end > x_start else 1
                actual_scale_y = self.target_size / h if h > 0 else 1
                
                metadata = {
                    'tile_id': idx,
                    'position': {
                        'x_start': x_start,
                        'x_end': x_end,
                        'y_start': 0,
                        'y_end': h
                    },
                    'scale': {
                        'x': actual_scale_x,
                        'y': actual_scale_y
                    },
                    'original_size': (w, h),
                    'tile_size': (self.target_size, self.target_size)
                }
            
            tiles.append((tile, metadata))
        
        return tiles
    
    def adaptive_resize_and_pad(self,
                                  image: np.ndarray,
                                  target_size: Tuple[int, int] = (640, 640),
                                  padding_value: int = 114) -> Tuple[np.ndarray, Dict]:
        """
        Resize image maintaining aspect ratio and pad to square
        
        This is an alternative to tiling when single-image processing is preferred
        
        Args:
            image: Input image
            target_size: Target (width, height)
            padding_value: Padding color (114 is standard for YOLO)
            
        Returns:
            Padded image and resize info dict
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scale to fit within target while preserving aspect ratio
        scale = min(target_w / w, target_h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded canvas
        padded = np.full((target_h, target_w, 3), padding_value, dtype=np.uint8)
        
        # Center the resized image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        metadata = {
            'original_size': (w, h),
            'resized_size': (new_w, new_h),
            'final_size': (target_w, target_h),
            'scale': scale,
            'padding': {
                'top': y_offset,
                'bottom': target_h - new_h - y_offset,
                'left': x_offset,
                'right': target_w - new_w - x_offset
            }
        }
        
        return padded, metadata
    
    def convert_yolo_boxes_for_tile(self, 
                                     original_boxes: List[List],
                                     tile_metadata: Dict) -> List[List]:
        """
        Convert YOLO format boxes from original image coordinates to tile coordinates
        
        Args:
            original_boxes: List of [class_id, x_center, y_center, w, h] in original image
            tile_metadata: Metadata from tile generation
            
        Returns:
            Converted boxes in tile coordinate system
        """
        converted_boxes = []
        
        orig_w, orig_h = tile_metadata['original_size']
        pos = tile_metadata['position']
        scale = tile_metadata['scale']
        
        x_start_norm = pos['x_start'] / orig_w
        x_end_norm = pos['x_end'] / orig_w
        
        for box in original_boxes:
            class_id, xc, yc, bw, bh = box
            
            # Check if box falls within this tile's horizontal range
            box_x_min = xc - bw/2
            box_x_max = xc + bw/2
            
            # Skip boxes not in this tile (with some tolerance)
            if box_x_max < x_start_norm or box_x_min > x_end_norm:
                continue
            
            # Convert to pixel coordinates in original image
            box_xc_px = xc * orig_w
            box_yc_px = yc * orig_h
            box_bw_px = bw * orig_w
            box_bh_px = bh * orig_h
            
            # Shift to tile coordinates
            tile_xc_px = box_xc_px - pos['x_start']
            
            # Scale to tile size
            tile_xc_norm = (tile_xc_px * scale['x']) / self.target_size
            tile_yc_norm = (box_yc_px * scale['y']) / self.target_size
            tile_bw_norm = (box_bw_px * scale['x']) / self.target_size
            tile_bh_norm = (box_bh_px * scale['y']) / self.target_size
            
            # Validate normalized coordinates
            if 0 <= tile_xc_norm <= 1 and 0 <= tile_yc_norm <= 1:
                converted_boxes.append([class_id, tile_xc_norm, tile_yc_norm, 
                                       min(tile_bw_norm, 1), min(tile_bh_norm, 1)])
        
        return converted_boxes
    
    def visualize_tiling_strategy(self, sample_image_path: str, output_path: str):
        """Create visualization showing how image is tiled"""
        
        img = cv2.imread(sample_image_path)
        if img is None:
            print(f"Cannot read image: {sample_image_path}")
            return
        
        # Create visualization canvas
        vis_img = img.copy()
        h, w = img.shape[:2]
        
        # Draw tile boundaries
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        
        tiles_data = self.tile_image_with_overlap(img)
        
        for idx, (tile_img, tile_meta) in enumerate(tiles_data[:len(colors)]):
            if tile_meta is None:
                continue
            
            pos = tile_meta['position']
            color = colors[idx % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(vis_img, 
                         (pos['x_start'], pos['y_start']),
                         (pos['x_end'], pos['y_end']),
                         color, 3)
            
            # Add label
            label = f"Tile {idx}"
            cv2.putText(vis_img, label,
                       (pos['x_start']+10, pos['y_start']+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Add title
        cv2.putText(vis_img, f"Tiling Strategy: {self.n_tiles_horizontal} tiles",
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(vis_img, f"Size: {w}x{h} -> {self.target_size}x{self.target_size}",
                   (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, vis_img)
        print(f"[SAVED] Tiling visualization: {output_path}")


def process_pfmd_dataset_for_training(input_dataset_dir: str,
                                       output_processed_dir: str,
                                       strategy: str = 'tiling',
                                       target_size: int = 640):
    """
    Process entire PF-MDD dataset using selected strategy
    
    Args:
        input_dataset_dir: Path to a YOLO-format dataset directory
        output_processed_dir: Output directory for processed data
        strategy: 'tiling' or 'adaptive_padding'
        target_size: Target image/tile size
    """
    
    input_dir = Path(input_dataset_dir)
    output_dir = Path(output_processed_dir)
    
    processor = WideImageProcessor(
        original_size=(4096, 256),
        target_tile_size=target_size,
        overlap_ratio=0.25
    )
    
    print("\n" + "="*70)
    print(f"PF-MDD DATASET PROCESSING")
    print(f"Strategy: {strategy.upper()}")
    print("="*70)
    
    # Create output structure
    splits = ['train', 'val', 'test']
    
    for split in splits:
        print(f"\n[Processing {split.upper()} set]")
        
        split_input = input_dir / "images" / split
        labels_input = input_dir / "labels" / split
        split_output_images = output_dir / "images" / split
        split_output_labels = output_dir / "labels" / split
        
        split_output_images.mkdir(parents=True, exist_ok=True)
        split_output_labels.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = sorted(list(split_input.glob("*.png")))
        total = len(image_files)
        
        print(f"  Found {total} images")
        
        processed_count = 0
        start_time = time.time()
        
        for img_idx, img_path in enumerate(image_files, 1):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            stem = img_path.stem
            
            # Load corresponding labels
            label_file = labels_input / f"{stem}.txt"
            original_boxes = []
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                box = [float(parts[0])] + [float(x) for x in parts[1:5]]
                                original_boxes.append(box)
                            except ValueError:
                                pass
            
            # Apply processing strategy
            if strategy == 'tiling':
                tiles_data = processor.tile_image_with_overlap(img)
                
                for tile_idx, (tile_img, tile_meta) in enumerate(tiles_data):
                    # Save tile image
                    tile_name = f"{stem}_t{tile_idx:02d}.png"
                    cv2.imwrite(str(split_output_images / tile_name), tile_img)
                    
                    # Convert and save labels for this tile
                    if tile_meta and original_boxes:
                        converted_boxes = processor.convert_yolo_boxes_for_tile(
                            original_boxes, tile_meta
                        )
                        
                        if converted_boxes:
                            label_out = split_output_labels / f"{stem}_t{tile_idx:02d}.txt"
                            with open(label_out, 'w') as f:
                                for box in converted_boxes:
                                    line = f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                                    f.write(line)
                    
                    processed_count += 1
                    
            elif strategy == 'adaptive_padding':
                padded_img, pad_meta = processor.adaptive_resize_and_pad(
                    img, (target_size, target_size)
                )
                
                # Save processed image
                out_name = f"{stem}_padded.png"
                cv2.imwrite(str(split_output_images / out_name), padded_img)
                
                # Save labels (coordinates remain valid due to normalization)
                if original_boxes:
                    label_out = split_output_labels / f"{stem}_padded.txt"
                    with open(label_out, 'w') as f:
                        for box in original_boxes:
                            line = f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                            f.write(line)
                
                processed_count += 1
            
            # Progress update
            if img_idx % 20 == 0 or img_idx == total:
                elapsed = time.time() - start_time
                print(f"    [{img_idx}/{total}] ({processed_count} tiles generated, {elapsed:.1f}s)")
        
        elapsed_split = time.time() - start_time
        print(f"\n  {split.upper()} complete:")
        print(f"    Images processed: {total}")
        print(f"    Total outputs: {processed_count}")
        print(f"    Time: {elapsed_split:.2f}s")
    
    # Generate summary report
    summary = {
        'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'strategy': strategy,
        'target_size': target_size,
        'original_resolution': [4096, 256],
        'tiles_per_image': processor.n_tiles_horizontal if strategy == 'tiling' else 1,
        'overlap_ratio': processor.overlap_ratio if strategy == 'tiling' else 0,
        'output_directory': str(output_dir.absolute())
    }
    
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("[PROCESSING COMPLETE]")
    print(f"Summary saved: {summary_path}")
    print(f"Output directory: {output_dir}")
    print('='*70)
    
    return summary


def main():
    """Demonstrate the image processing strategies"""
    
    base_dir = BASE_DIR
    dataset_dir = preferred_training_dataset_dir()
    output_dir = base_dir / "pfmd_processed_for_yolo"
    
    print("\n" + "#"*70)
    print("# PF-MDD Wide-Format Image Processing Demo")
    print("#"*70)
    
    # Show configuration
    processor = WideImageProcessor(
        original_size=(4096, 256),
        target_tile_size=640,
        overlap_ratio=0.25
    )
    
    # Visualize on a sample image
    sample_img = dataset_dir / "images/train"
    if sample_img.exists():
        sample_files = list(sample_img.glob("*.png"))
        if sample_files:
            vis_output = base_dir / "tiling_visualization.png"
            processor.visualize_tiling_strategy(str(sample_files[0]), str(vis_output))
    
    # Ask user which strategy to use for full dataset
    print("""
Available Strategies:

1. TILING (Recommended for PF-MDD)
   - Splits 4096x256 into ~7 overlapping 640x640 tiles
   - Preserves defect details at original resolution
   - Better detection of small defects
   - Increases dataset size ~7x

2. ADAPTIVE PADDING
   - Resizes to 640x640 with letterboxing
   - Simpler but may lose fine details
   - Faster processing
   - Maintains original dataset size

For baseline training, RECOMMENDED: Use TILING strategy
This will give best performance on small defect types.
""")
    
    return True


if __name__ == "__main__":
    main()
