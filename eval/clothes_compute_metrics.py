"""
Batch compute metrics for garment pattern prediction.
Supports V3 Lite and V4 JSON formats.

Usage:
    python compute_metrics_batch.py --base_dir /path/to/results
    python compute_metrics_batch.py --base_dir /path/to/data --gt_dir json --pred_dir json_v4

Expected directory structure:
    base_dir/
        <gt_dir>/         (default: ground_truth)
            sample1.json  (V3 Lite or V4 format)
            ...
        <pred_dir>/       (default: json)
            sample1.json  (V3 Lite or V4 format)
            ...
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from shapely.affinity import translate
from tqdm import tqdm
import argparse


# ============================================================================
# Decoder - Supports V3 Lite and V4 formats
# ============================================================================

class DecoderV3LiteV4:
    """Decoder for V3 Lite and V4 formats (compatible)"""
    
    def __init__(self):
        self.template_w = 640.0
    
    def decode(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decode V3 Lite or V4 to specification.json format"""
        
        meta = data.get("meta", {})
        panels_data = data.get("panels", [])
        stitches_data = data.get("stitches", [])
        
        scale = meta.get("scale", 2.75)
        
        # Decode panels
        panels = {}
        panel_id_to_name = {}
        
        for panel_data in panels_data:
            pid = panel_data.get("panel_id", 0)
            panel_name = f"panel_{pid}"
            panel_id_to_name[pid] = panel_name
            
            panel = self._decode_panel(panel_data, scale)
            panels[panel_name] = panel
        
        # Decode stitches
        stitches = []
        for stitch_data in stitches_data:
            stitch = self._decode_stitch(stitch_data, panel_id_to_name)
            if stitch:
                stitches.append(stitch)
        
        return {
            "parameters": {},
            "parameter_order": [],
            "pattern": {
                "panels": panels,
                "stitches": stitches,
            },
            "properties": {
                "curvature_coords": "relative",
                "normalize_panel_translation": False,
                "normalized_edge_loops": True,
                "units_in_meter": 100,
            },
        }
    
    def _decode_panel(self, panel_data: Dict, scale: float) -> Dict:
        """Decode single panel"""
        
        label = panel_data.get("panel_label", "")
        side = panel_data.get("side", "front")
        
        tx = panel_data.get("translation_x", 0)
        ty = panel_data.get("translation_y", 0)
        tz = panel_data.get("translation_z", 30.0 if side == "front" else -30.0)
        
        rx = panel_data.get("rotation_x", 0)
        ry = panel_data.get("rotation_y", 0)
        rz = panel_data.get("rotation_z", 0)
        
        # Vertices (inverse scaling)
        scaled_vertices = panel_data.get("vertices", [])
        vertices = []
        for sv in scaled_vertices:
            v = [sv[0] / (100 * scale), sv[1] / (100 * scale)]
            vertices.append(v)
        
        # Edges
        encoded_edges = panel_data.get("edges", [])
        edges = self._decode_edges(encoded_edges, len(vertices))
        
        # Translation (inverse calculation)
        trans_x = tx * self.template_w / (100 * scale)
        trans_y = ty * self.template_w / (100 * scale)
        trans_z = tz
        
        return {
            "vertices": vertices,
            "edges": edges,
            "label": label,
            "translation": [trans_x, trans_y, trans_z],
            "rotation": [rx, ry, rz],
        }
    
    def _decode_edges(self, encoded_edges: List, num_vertices: int) -> List[Dict]:
        """Decode edges (supports both V3 Lite and V4 formats)"""
        edges = []
        
        # V4 format uses edge_index, V3 Lite uses position
        # Build a mapping from edge_index to edge_data
        edge_map = {}
        for edge_data in encoded_edges:
            idx = edge_data.get("edge_index", len(edge_map))
            edge_map[idx] = edge_data
        
        for i in range(num_vertices):
            edge = {
                "endpoints": [i, (i + 1) % num_vertices]
            }
            
            edge_data = edge_map.get(i, {})
            curve_type = edge_data.get("curve_type", "line")
            curve_params = edge_data.get("curve_params", [])
            
            if curve_type == "line":
                pass
            elif curve_type == "quadratic":
                if len(curve_params) >= 2:
                    edge["curvature"] = [curve_params[0], curve_params[1]]
            elif curve_type == "cubic":
                if len(curve_params) >= 4:
                    edge["curvature"] = {
                        "type": "cubic",
                        "params": [
                            [curve_params[0], curve_params[1]],
                            [curve_params[2], curve_params[3]],
                        ]
                    }
            elif curve_type == "circle":
                if len(curve_params) >= 3:
                    edge["curvature"] = {
                        "type": "circle",
                        "params": [
                            curve_params[0],
                            curve_params[1],
                            curve_params[2]
                        ]
                    }
            
            edges.append(edge)
        
        return edges
    
    def _decode_stitch(self, stitch_data: Dict, panel_id_to_name: Dict) -> Optional[List[Dict]]:
        """Decode stitch"""
        
        p1_id = stitch_data.get("from_panel_id")
        e1_idx = stitch_data.get("from_edge_index")
        p2_id = stitch_data.get("to_panel_id")
        e2_idx = stitch_data.get("to_edge_index")
        
        if p1_id is None or p2_id is None:
            return None
        
        p1_name = panel_id_to_name.get(p1_id)
        p2_name = panel_id_to_name.get(p2_id)
        
        if not p1_name or not p2_name:
            return None
        
        return [
            {"panel": p1_name, "edge": e1_idx},
            {"panel": p2_name, "edge": e2_idx},
        ]


# ============================================================================
# Metrics computation functions
# ============================================================================


def sample_edge_points(v0: np.ndarray, v1: np.ndarray, curvature: Optional[dict] = None, num_samples: int = 10) -> np.ndarray:
    """Sample points along an edge."""
    t = np.linspace(0, 1, num_samples)
    points = v0[None, :] + t[:, None] * (v1 - v0)[None, :]
    return points


def sample_panel_boundary(panel: dict, num_samples_per_edge: int = 10) -> np.ndarray:
    """Sample points along the boundary of a panel."""
    vertices = np.array(panel['vertices'])
    edges = panel.get('edges', [])
    
    if len(vertices) < 2:
        return vertices
    
    all_points = []
    for edge in edges:
        endpoints = edge.get('endpoints', [])
        if len(endpoints) < 2:
            continue
        v0_idx, v1_idx = endpoints
        if v0_idx >= len(vertices) or v1_idx >= len(vertices):
            continue
        v0 = vertices[v0_idx]
        v1 = vertices[v1_idx]
        
        points = sample_edge_points(v0, v1, None, num_samples_per_edge)
        all_points.append(points[:-1])
    
    if not all_points:
        return vertices
    
    return np.vstack(all_points)


def align_points_centroid(points1: np.ndarray, points2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align two point sets by centering them at origin."""
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)
    return points1 - centroid1, points2 - centroid2


def chamfer_distance_2d(points1: np.ndarray, points2: np.ndarray, align: bool = True) -> float:
    """Compute bidirectional Chamfer distance between two point sets."""
    if align:
        points1, points2 = align_points_centroid(points1, points2)
    
    dist_matrix = cdist(points1, points2)
    forward = np.mean(np.min(dist_matrix, axis=1))
    backward = np.mean(np.min(dist_matrix, axis=0))
    
    return (forward + backward) / 2


def get_panel_polygon(panel: dict) -> Optional[Polygon]:
    """Convert panel to Shapely polygon for IoU computation."""
    vertices = np.array(panel['vertices'])
    edges = panel['edges']
    
    if len(vertices) < 3:
        return None
    
    boundary_points = []
    for edge in edges:
        v0_idx, v1_idx = edge['endpoints']
        v0 = vertices[v0_idx]
        v1 = vertices[v1_idx]
        curvature = edge.get('curvature', None)
        
        points = sample_edge_points(v0, v1, curvature, num_samples=20)
        boundary_points.extend(points[:-1].tolist())
    
    if len(boundary_points) < 3:
        return None
    
    try:
        poly = Polygon(boundary_points)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except Exception:
        return None


def compute_panel_iou(pred_panel: dict, gt_panel: dict, align: bool = True) -> float:
    """Compute IoU between two panels."""
    pred_poly = get_panel_polygon(pred_panel)
    gt_poly = get_panel_polygon(gt_panel)
    
    if pred_poly is None or gt_poly is None:
        return 0.0
    
    if pred_poly.is_empty or gt_poly.is_empty:
        return 0.0
    
    if align:
        try:
            pred_centroid = pred_poly.centroid
            gt_centroid = gt_poly.centroid
            if pred_centroid.is_empty or gt_centroid.is_empty:
                return 0.0
            pred_poly = translate(pred_poly, -pred_centroid.x, -pred_centroid.y)
            gt_poly = translate(gt_poly, -gt_centroid.x, -gt_centroid.y)
        except Exception:
            return 0.0
    
    try:
        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        
        if union == 0:
            return 0.0
        return intersection / union
    except Exception:
        return 0.0


def decode_v3_lite(v3_lite: dict) -> dict:
    """Decode V3 Lite or V4 format to specification format."""
    decoder = DecoderV3LiteV4()
    return decoder.decode(v3_lite)


# Alias for V4
decode_v4 = decode_v3_lite


def compute_2d_chamfer(pred_spec: dict, gt_spec: dict, num_samples_per_edge: int = 20) -> Tuple[float, dict]:
    """Compute 2D Chamfer distance with centroid alignment."""
    pred_panels = pred_spec['pattern']['panels']
    gt_panels = gt_spec['pattern']['panels']
    
    common_panels = set(pred_panels.keys()) & set(gt_panels.keys())
    
    per_panel_chamfer = {}
    total_chamfer = 0.0
    
    for panel_name in common_panels:
        pred_points = sample_panel_boundary(pred_panels[panel_name], num_samples_per_edge)
        gt_points = sample_panel_boundary(gt_panels[panel_name], num_samples_per_edge)
        
        # Chamfer distance in cm, then convert to mm
        chamfer_cm = chamfer_distance_2d(pred_points, gt_points, align=True)
        chamfer_mm = chamfer_cm * 10
        
        per_panel_chamfer[panel_name] = chamfer_mm
        total_chamfer += chamfer_mm
    
    avg_chamfer = total_chamfer / len(common_panels) if common_panels else 0.0
    
    return avg_chamfer, per_panel_chamfer


def compute_2d_iou(pred_spec: dict, gt_spec: dict) -> Tuple[float, dict]:
    """Compute 2D IoU with centroid alignment."""
    pred_panels = pred_spec['pattern']['panels']
    gt_panels = gt_spec['pattern']['panels']
    
    common_panels = set(pred_panels.keys()) & set(gt_panels.keys())
    
    per_panel_iou = {}
    total_iou = 0.0
    valid_count = 0
    
    for panel_name in common_panels:
        iou = compute_panel_iou(pred_panels[panel_name], gt_panels[panel_name], align=True)
        if iou > 0:
            per_panel_iou[panel_name] = iou * 100
            total_iou += iou
            valid_count += 1
    
    avg_iou = (total_iou / valid_count * 100) if valid_count > 0 else 0.0
    
    return avg_iou, per_panel_iou


def get_edge_geometry(panel: dict, edge_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Get edge geometry: midpoint, direction, and length."""
    vertices = np.array(panel['vertices'])
    edges = panel['edges']
    
    if edge_idx >= len(edges):
        return None, None, 0
    
    edge = edges[edge_idx]
    v0_idx, v1_idx = edge['endpoints']
    v0 = vertices[v0_idx]
    v1 = vertices[v1_idx]
    
    midpoint = (v0 + v1) / 2
    direction = v1 - v0
    length = np.linalg.norm(direction)
    if length > 0:
        direction = direction / length
    
    return midpoint, direction, length


def stitch_to_geometry(stitch: List[dict], panels: dict) -> Tuple:
    """Convert stitch to geometry representation."""
    results = []
    for item in stitch:
        panel_name = item['panel']
        edge_idx = item['edge']
        
        if panel_name not in panels:
            return None
        
        panel = panels[panel_name]
        midpoint, direction, length = get_edge_geometry(panel, edge_idx)
        
        if midpoint is None:
            return None
        
        results.append((panel_name, midpoint, length))
    
    results = sorted(results, key=lambda x: x[0])
    return tuple(results)


def match_stitch_by_geometry(pred_stitch_geo, gt_stitch_geo, 
                              distance_threshold: float = 8.0, 
                              length_ratio_threshold: float = 0.6) -> bool:
    """Check if two stitches match based on geometry."""
    if pred_stitch_geo is None or gt_stitch_geo is None:
        return False
    
    pred_panels = set(item[0] for item in pred_stitch_geo)
    gt_panels = set(item[0] for item in gt_stitch_geo)
    
    if pred_panels != gt_panels:
        return False
    
    for pred_item in pred_stitch_geo:
        panel_name = pred_item[0]
        pred_midpoint = pred_item[1]
        pred_length = pred_item[2]
        
        gt_item = next((g for g in gt_stitch_geo if g[0] == panel_name), None)
        if gt_item is None:
            return False
        
        gt_midpoint = gt_item[1]
        gt_length = gt_item[2]
        
        dist = np.linalg.norm(pred_midpoint - gt_midpoint)
        if dist > distance_threshold:
            return False
        
        if gt_length > 0:
            length_diff = abs(pred_length - gt_length) / gt_length
            if length_diff > length_ratio_threshold:
                return False
    
    return True


def compute_stitch_accuracy(pred_spec: dict, gt_spec: dict) -> Tuple[float, dict]:
    """Compute stitch connection accuracy using geometric matching."""
    pred_stitches = pred_spec['pattern'].get('stitches', [])
    gt_stitches = gt_spec['pattern'].get('stitches', [])
    pred_panels = pred_spec['pattern']['panels']
    gt_panels = gt_spec['pattern']['panels']
    
    if not pred_stitches or not gt_stitches:
        return 0.0, {'correct': 0, 'total_gt': len(gt_stitches), 'total_pred': len(pred_stitches)}
    
    pred_geos = [stitch_to_geometry(s, pred_panels) for s in pred_stitches]
    gt_geos = [stitch_to_geometry(s, gt_panels) for s in gt_stitches]
    
    pred_geos = [g for g in pred_geos if g is not None]
    gt_geos = [g for g in gt_geos if g is not None]
    
    matched_gt = set()
    correct = 0
    
    for pred_geo in pred_geos:
        for i, gt_geo in enumerate(gt_geos):
            if i not in matched_gt and match_stitch_by_geometry(pred_geo, gt_geo):
                correct += 1
                matched_gt.add(i)
                break
    
    total_gt = len(gt_geos)
    total_pred = len(pred_geos)
    
    accuracy = (correct / total_gt * 100) if total_gt > 0 else 0.0
    precision = (correct / total_pred * 100) if total_pred > 0 else 0.0
    
    stats = {
        'correct': correct,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'accuracy': accuracy,
        'precision': precision,
    }
    
    return accuracy, stats


def compute_metrics_for_pair(pred_path: Path, gt_path: Path) -> dict:
    """Compute metrics for a single pred-gt pair."""
    results = {
        'sample_name': pred_path.stem,
        '2d_chamfer_mm': None,
        '2d_iou': None,
        'stitch_accuracy': None,
        'panel_count_match': False,
    }
    
    try:
        with open(pred_path) as f:
            pred_v3lite = json.load(f)
        with open(gt_path) as f:
            gt_v3lite = json.load(f)
    except Exception as e:
        print(f"Failed to load {pred_path.name}: {e}")
        return results
    
    # Decode V3 Lite format to specification format
    try:
        pred_spec = decode_v3_lite(pred_v3lite)
        gt_spec = decode_v3_lite(gt_v3lite)
    except Exception as e:
        print(f"Failed to decode {pred_path.name}: {e}")
        return results
    
    # Check if pattern exists
    if 'pattern' not in pred_spec or 'pattern' not in gt_spec:
        return results
    
    if 'panels' not in pred_spec['pattern'] or 'panels' not in gt_spec['pattern']:
        return results
    
    # Panel count match
    pred_panels = set(pred_spec['pattern']['panels'].keys())
    gt_panels = set(gt_spec['pattern']['panels'].keys())
    results['panel_count_match'] = (pred_panels == gt_panels)
    results['pred_panel_count'] = len(pred_panels)
    results['gt_panel_count'] = len(gt_panels)
    
    # 2D Chamfer
    chamfer_2d, _ = compute_2d_chamfer(pred_spec, gt_spec)
    results['2d_chamfer_mm'] = chamfer_2d
    
    # 2D IoU
    iou_2d, _ = compute_2d_iou(pred_spec, gt_spec)
    results['2d_iou'] = iou_2d
    
    # Stitch Accuracy
    stitch_acc, stitch_stats = compute_stitch_accuracy(pred_spec, gt_spec)
    results['stitch_accuracy'] = stitch_acc
    results['stitch_stats'] = stitch_stats
    
    return results


def compute_metrics_batch(base_dir: str, gt_subdir: str = "ground_truth", pred_subdir: str = "json") -> dict:
    """Compute metrics for all samples in the directory.
    
    Args:
        base_dir: Base directory path
        gt_subdir: Subdirectory name for ground truth (default: "ground_truth")
        pred_subdir: Subdirectory name for predictions (default: "json")
    """
    base_path = Path(base_dir)
    gt_dir = base_path / gt_subdir
    pred_dir = base_path / pred_subdir
    
    if not gt_dir.exists() or not pred_dir.exists():
        raise ValueError(f"Missing {gt_subdir} or {pred_subdir} directory in {base_dir}")
    
    # Find matching files
    gt_files = {f.stem: f for f in gt_dir.glob("*.json")}
    pred_files = {f.stem: f for f in pred_dir.glob("*.json")}
    
    common_samples = set(gt_files.keys()) & set(pred_files.keys())
    print(f"Found {len(common_samples)} matching samples")
    print(f"  GT only: {len(gt_files) - len(common_samples)}")
    print(f"  Pred only: {len(pred_files) - len(common_samples)}")
    
    # Compute metrics for each pair
    all_results = []
    for sample_name in tqdm(sorted(common_samples), desc="Computing metrics"):
        results = compute_metrics_for_pair(pred_files[sample_name], gt_files[sample_name])
        all_results.append(results)
    
    # Aggregate results
    valid_chamfer = [r for r in all_results if r['2d_chamfer_mm'] is not None]
    valid_iou = [r for r in all_results if r['2d_iou'] is not None]
    valid_stitch = [r for r in all_results if r['stitch_accuracy'] is not None]
    
    summary = {
        'num_samples': len(all_results),
        'num_valid': len(valid_chamfer),
        'panel_count_match_rate': np.mean([r['panel_count_match'] for r in all_results]) * 100,
        'avg_2d_chamfer_mm': np.mean([r['2d_chamfer_mm'] for r in valid_chamfer]) if valid_chamfer else None,
        'std_2d_chamfer_mm': np.std([r['2d_chamfer_mm'] for r in valid_chamfer]) if valid_chamfer else None,
        'avg_2d_iou': np.mean([r['2d_iou'] for r in valid_iou]) if valid_iou else None,
        'std_2d_iou': np.std([r['2d_iou'] for r in valid_iou]) if valid_iou else None,
        'avg_stitch_accuracy': np.mean([r['stitch_accuracy'] for r in valid_stitch]) if valid_stitch else None,
        'std_stitch_accuracy': np.std([r['stitch_accuracy'] for r in valid_stitch]) if valid_stitch else None,
    }
    
    return {
        'summary': summary,
        'per_sample': all_results
    }


def main():
    parser = argparse.ArgumentParser(description='Batch compute metrics for garment patterns (V3 Lite / V4)')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Path to base directory')
    parser.add_argument('--gt_dir', type=str, default='ground_truth',
                        help='Subdirectory name for ground truth (default: ground_truth)')
    parser.add_argument('--pred_dir', type=str, default='json',
                        help='Subdirectory name for predictions (default: json)')
    parser.add_argument('--output', type=str, default='batch_metrics_results.json',
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    results = compute_metrics_batch(args.base_dir, args.gt_dir, args.pred_dir)
    summary = results['summary']
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total samples: {summary['num_samples']}")
    print(f"Valid samples: {summary['num_valid']}")
    print(f"Panel count match rate: {summary['panel_count_match_rate']:.2f}%")
    print()
    print("2D Pattern Quality:")
    if summary['avg_2d_chamfer_mm'] is not None:
        print(f"  2D Chamfer (mm): {summary['avg_2d_chamfer_mm']:.4f} +/- {summary['std_2d_chamfer_mm']:.4f}")
    if summary['avg_2d_iou'] is not None:
        print(f"  2D IoU (%): {summary['avg_2d_iou']:.2f} +/- {summary['std_2d_iou']:.2f}")
    if summary['avg_stitch_accuracy'] is not None:
        print(f"  Stitch Accuracy (%): {summary['avg_stitch_accuracy']:.2f} +/- {summary['std_stitch_accuracy']:.2f}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()

