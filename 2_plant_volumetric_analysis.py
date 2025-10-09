# Developed: Worasit Sangjan
# Date 9 October 2025

import os
import csv
import glob
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

# ============================================================================
# CONFIGURATION (Optional: Edit these defaults)
# ============================================================================

DEFAULT_INPUT_DIR = "individual_plants"
DEFAULT_OUTPUT_DIR = "volumetric_results"
DEFAULT_VOXEL_SIZE_MM = 5.0
DEFAULT_INPUT_UNIT = "mm"
DEFAULT_DILATE_ITERS = 1

# ============================================================================
# PART 1: CLEANING
# ============================================================================

def clean_plant_file(pcd, eps=20.0, min_samples=50, min_cluster_ratio=0.05,
                     remove_non_green=True, green_threshold=110, 
                     green_dominance=1.1, verbose=True):
    """Clean individual plant using color filter and spatial clustering."""
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    original_count = len(pts)

    if verbose:
        print(f"    Original points: {original_count:,}")

    # Color filtering (optional)
    if remove_non_green and cols is not None and len(cols) == len(pts):
        cols255 = (cols * 255.0).astype(np.float32)
        R, G, B = cols255[:, 0], cols255[:, 1], cols255[:, 2]
        
        green_mask = (G > green_threshold) & (G > R * green_dominance) & (G > B * green_dominance)
        yellow_mask = (G > 100) & (R > 100) & (G > B * 1.2)
        keep_color = green_mask | yellow_mask

        pts_f = pts[keep_color]
        cols_f = cols[keep_color]
        
        if verbose:
            print(f"    Color filter: {len(pts_f):,} kept")
    else:
        pts_f = pts
        cols_f = cols

    # Spatial clustering (DBSCAN)
    if len(pts_f) > 0:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts_f)
        labels = db.labels_
        
        # Keep only sufficiently large clusters
        mask_valid = labels >= 0
        uniq, counts = np.unique(labels[mask_valid], return_counts=True)
        
        if len(uniq) > 0:
            max_size = counts.max()
            size_thresh = max_size * float(min_cluster_ratio)
            big_clusters = uniq[counts >= size_thresh]
            keep = np.isin(labels, big_clusters)
            pts_c = pts_f[keep]
            cols_c = cols_f[keep] if cols_f is not None else None
            
            if verbose:
                print(f"    Spatial: {len(pts_c):,} points ({len(big_clusters)} clusters)")
        else:
            pts_c = pts_f
            cols_c = cols_f
    else:
        pts_c = pts_f
        cols_c = cols_f

    if verbose:
        pct = 100.0 * (len(pts_c) / max(1, original_count))
        print(f"    Final: {len(pts_c):,} points ({pct:.1f}%)")

    # Create output point cloud
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts_c)
    if cols_c is not None:
        out.colors = o3d.utility.Vector3dVector(cols_c)
    
    return out

# ============================================================================
# PART 2: VOXEL UTILITIES
# ============================================================================

def scale_units(points, unit):
    """Scale points to meters based on input unit."""
    lut = {"m": 1.0, "meter": 1.0, "meters": 1.0, "cm": 0.01, "mm": 0.001, "millimeter": 0.001}
    unit = (unit or "m").lower()
    if unit not in lut:
        raise ValueError(f"Unknown unit '{unit}'. Use m|cm|mm.")
    return points * lut[unit]

def voxelize(points_m, voxel):
    """Convert points (meters) to occupancy grid."""
    mn = points_m.min(axis=0)
    idx = np.floor((points_m - mn) / voxel).astype(np.int64)
    shape = idx.max(axis=0) + 1
    occ = np.zeros(shape, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ, mn, voxel

def dilate6(occ, iters=1):
    """6-neighborhood dilation (fills tiny gaps)."""
    if iters <= 0:
        return occ
    cur = occ.copy()
    for _ in range(iters):
        x0 = np.pad(cur, ((1,0),(0,0),(0,0)), "constant")[:-1,:,:]
        x1 = np.pad(cur, ((0,1),(0,0),(0,0)), "constant")[1:,:,:]
        y0 = np.pad(cur, ((0,0),(1,0),(0,0)), "constant")[:,:-1,:]
        y1 = np.pad(cur, ((0,0),(0,1),(0,0)), "constant")[:,1:,:]
        z0 = np.pad(cur, ((0,0),(0,0),(1,0)), "constant")[:,:,:-1]
        z1 = np.pad(cur, ((0,0),(0,0),(0,1)), "constant")[:,:,1:]
        cur = cur | x0 | x1 | y0 | y1 | z0 | z1
    return cur

def surface_area_from_occupancy(occ, voxel):
    """Face count method for surface area (m²)."""
    v2 = voxel * voxel
    area = 0.0
    area += np.sum(occ[1:, :, :] & ~occ[:-1, :, :]) * v2
    area += np.sum(occ[:-1, :, :] & ~occ[1:, :, :]) * v2
    area += np.sum(occ[:, 1:, :] & ~occ[:, :-1, :]) * v2
    area += np.sum(occ[:, :-1, :] & ~occ[:, 1:, :]) * v2
    area += np.sum(occ[:, :, 1:] & ~occ[:, :, :-1]) * v2
    area += np.sum(occ[:, :, :-1] & ~occ[:, :, 1:]) * v2
    return float(area)

def voxel_surface_indices(occ):
    """Return (i,j,k) indices of surface voxels (at least one empty 6-neighbor)."""
    pad = np.pad(occ, 1, "constant")
    cx = pad[1:-1,1:-1,1:-1]
    neighbors = (
        pad[2:,1:-1,1:-1] & pad[:-2,1:-1,1:-1] &
        pad[1:-1,2:,1:-1] & pad[1:-1,:-2,1:-1] &
        pad[1:-1,1:-1,2:] & pad[1:-1,1:-1,:-2]
    )
    return np.argwhere(cx & (~neighbors))

# ============================================================================
# PART 3: EXPORT FUNCTIONS
# ============================================================================

def write_obj_voxel_cubes(path, indices, origin, voxel_size, max_cubes=50000):
    """Write an OBJ with cubes at the given surface voxel indices."""
    indices = np.asarray(indices)
    if indices.shape[0] == 0:
        with open(path, "w") as f:
            f.write("# empty voxel mesh\n")
        return
    
    if indices.shape[0] > max_cubes:
        sel = np.random.choice(indices.shape[0], size=max_cubes, replace=False)
        indices = indices[sel]

    base_v = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                       [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=float)
    base_f = np.array([
        [1,2,3],[1,3,4], [5,8,7],[5,7,6],
        [1,5,6],[1,6,2], [2,6,7],[2,7,3],
        [3,7,8],[3,8,4], [4,8,5],[4,5,1],
    ], dtype=int)

    with open(path, "w") as f:
        f.write("# voxel cubes\n")
        vcount = 0
        for (i,j,k) in indices:
            lower = origin + np.array([i,j,k], float) * voxel_size
            V = lower + base_v * voxel_size
            for x,y,z in V:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for a,b,c in base_f:
                f.write(f"f {vcount+a} {vcount+b} {vcount+c}\n")
            vcount += 8

def write_ply_points_ascii(path, xyz):
    """Write ASCII PLY with just XYZ points."""
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, xyz, fmt="%.6f")

# ============================================================================
# PART 4: PROCESSING PIPELINE
# ============================================================================

def process_plant_voxel(input_file, output_dir, plant_name, 
                       input_unit="mm", voxel_size_mm=5.0, dilate_iters=1,
                       clean_params=None, verbose=True):
    """Run cleaning + voxel metrics + visual exports for one plant."""
    os.makedirs(output_dir, exist_ok=True)

    if clean_params is None:
        clean_params = {
            "eps": 20.0,
            "min_samples": 50,
            "min_cluster_ratio": 0.05,
            "remove_non_green": True,
            "green_threshold": 110,
            "green_dominance": 1.1,
        }

    if verbose:
        print("\n" + "="*70)
        print(plant_name)
        print("="*70)
        print("  CLEANING:")

    # Load original cloud
    pcd = o3d.io.read_point_cloud(input_file)

    # Clean (DBSCAN eps is in original file units)
    pcd_clean = clean_plant_file(pcd, verbose=verbose, **clean_params)

    # Save cleaned cloud
    clean_path = os.path.join(output_dir, f"{plant_name}_cleaned.ply")
    o3d.io.write_point_cloud(clean_path, pcd_clean)
    if verbose:
        print(f"    Saved: {clean_path}")

    # Points to meters
    pts = np.asarray(pcd_clean.points)
    pts_m = scale_units(pts, input_unit)

    # Stats
    mn, mx = pts_m.min(axis=0), pts_m.max(axis=0)
    extent = mx - mn
    if verbose:
        print("\n  MEASUREMENTS:")
        print(f"    Points: {len(pts):,}")
        print(f"    Extent: {(extent*1000).round(2)} mm")
        print(f"    Height: {extent[2]*1000:.1f} mm")

    # Resolutions
    voxel_sizes = {
        "coarse": 2.0 * voxel_size_mm / 1000.0,
        "medium": 1.0 * voxel_size_mm / 1000.0,
        "fine":   0.5 * voxel_size_mm / 1000.0,
    }

    if verbose:
        print("\n  VOXEL ANALYSIS:")

    results = {"plant": plant_name, "points": int(len(pts))}
    
    for name, voxel_m in voxel_sizes.items():
        try:
            if verbose:
                print(f"    {name.capitalize()} ({voxel_m*1000:.1f} mm)...", end="")
            
            # Occupancy
            occ, origin, _ = voxelize(pts_m, voxel_m)
            if dilate_iters > 0:
                occ = dilate6(occ, iters=dilate_iters)

            # Metrics
            occupied = int(occ.sum())
            vol_m3 = occupied * (voxel_m ** 3)
            area_m2 = surface_area_from_occupancy(occ, voxel_m)
            results[f"{name}_voxels"] = occupied
            results[f"{name}_volume_cm3"] = vol_m3 * 1e6
            results[f"{name}_area_cm2"]   = area_m2 * 1e4

            # Visual exports
            mm = int(round(voxel_m * 1000))
            surf_idx = voxel_surface_indices(occ)
            centers = (surf_idx + 0.5) * voxel_m + origin

            stem = os.path.join(output_dir, f"{plant_name}_{name}_voxels_{mm}mm")
            obj_path = stem + ".obj"
            ply_path = stem + ".ply"

            write_obj_voxel_cubes(obj_path, surf_idx, origin, voxel_m, max_cubes=50000)
            write_ply_points_ascii(ply_path, centers)

            if verbose:
                print(f" Vol={vol_m3*1e6:.1f} cm³, Area={area_m2*1e4:.1f} cm², Vox={occupied:,}")
        
        except Exception as e:
            if verbose:
                print(f" FAILED: {e}")
            results[f"{name}_voxels"] = 0
            results[f"{name}_volume_cm3"] = 0.0
            results[f"{name}_area_cm2"]   = 0.0

    return results

def process_all_plants(input_dir="individual_plants", output_dir="individual_plants_voxel",
                      output_csv="plant_measurements.csv", input_unit="mm",
                      voxel_size_mm=5.0, dilate_iters=1):
    """Batch over a folder of .ply plants."""
    print("="*70)
    print("PLANT VOLUMETRIC ANALYSIS - VOXEL METHOD")
    print("="*70)
    print(f"Input:  {input_dir}/  (units: {input_unit})")
    print(f"Output: {output_dir}/")
    print(f"Voxel sizes: {voxel_size_mm*2:.1f}mm, {voxel_size_mm:.1f}mm, {voxel_size_mm*0.5:.1f}mm")
    print(f"Dilation: {dilate_iters} iteration(s)")
    os.makedirs(output_dir, exist_ok=True)

    ply_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".ply")])
    if not ply_files:
        print(f"\nERROR: No PLY files in {input_dir}/")
        return []

    print(f"\nProcessing {len(ply_files)} plant(s)...\n")
    all_results = []
    
    for fname in ply_files:
        plant = os.path.splitext(fname)[0]
        in_path = os.path.join(input_dir, fname)
        res = process_plant_voxel(
            in_path, output_dir, plant,
            input_unit=input_unit,
            voxel_size_mm=voxel_size_mm,
            dilate_iters=dilate_iters,
            verbose=True,
        )
        all_results.append(res)

    # Save CSV
    csv_path = os.path.join(output_dir, output_csv)
    print("\n" + "="*70)
    print("SAVING MEASUREMENTS")
    print("="*70)
    
    fields = [
        "plant", "points",
        "coarse_volume_cm3", "coarse_area_cm2", "coarse_voxels",
        "medium_volume_cm3", "medium_area_cm2", "medium_voxels",
        "fine_volume_cm3",   "fine_area_cm2",   "fine_voxels",
    ]
    
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            row = {k: r.get(k, 0) for k in fields}
            w.writerow(row)
    
    print(f"Saved: {csv_path}")

    # Summary
    print("\nVOLUME SUMMARY (cm³):")
    print("-"*70)
    print(f"{'Plant':<20}{'Points':>10}{'Coarse':>12}{'Medium':>12}{'Fine':>12}")
    print("-"*70)
    for r in all_results:
        print(f"{r['plant']:<20}{r['points']:>10,}"
              f"{r.get('coarse_volume_cm3',0):>12.1f}"
              f"{r.get('medium_volume_cm3',0):>12.1f}"
              f"{r.get('fine_volume_cm3',0):>12.1f}")
    
    print("\n" + "="*70)
    print("VOLUMETRIC ANALYSIS COMPLETE")
    print("="*70)
    return all_results

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def get_user_inputs():
    """Interactively prompt user for configuration."""
    print("="*70)
    print("PLANT VOLUMETRIC ANALYSIS")
    print("="*70)
    
    input_dir = input(f"\nInput directory with plant PLY files [{DEFAULT_INPUT_DIR}]: ").strip()
    if not input_dir:
        input_dir = DEFAULT_INPUT_DIR
    
    output_dir = input(f"Output directory [{DEFAULT_OUTPUT_DIR}]: ").strip()
    if not output_dir:
        output_dir = DEFAULT_OUTPUT_DIR
    
    unit_input = input(f"Input units (m/cm/mm) [{DEFAULT_INPUT_UNIT}]: ").strip()
    input_unit = unit_input if unit_input else DEFAULT_INPUT_UNIT
    
    voxel_input = input(f"Base voxel size in mm [{DEFAULT_VOXEL_SIZE_MM}]: ").strip()
    voxel_size_mm = float(voxel_input) if voxel_input else DEFAULT_VOXEL_SIZE_MM
    
    dilate_input = input(f"Dilation iterations [{DEFAULT_DILATE_ITERS}]: ").strip()
    dilate_iters = int(dilate_input) if dilate_input else DEFAULT_DILATE_ITERS
    
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"Input dir:     {input_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Input units:   {input_unit}")
    print(f"Voxel sizes:   {voxel_size_mm*2:.1f}mm, {voxel_size_mm:.1f}mm, {voxel_size_mm*0.5:.1f}mm")
    print(f"Dilation:      {dilate_iters} iteration(s)")
    print("="*70)
    
    input("\nPress Enter to start processing...")
    
    return {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'input_unit': input_unit,
        'voxel_size_mm': voxel_size_mm,
        'dilate_iters': dilate_iters
    }

def main():
    """Main entry point."""
    config = get_user_inputs()
    
    process_all_plants(
        input_dir=config['input_dir'],
        output_dir=config['output_dir'],
        input_unit=config['input_unit'],
        voxel_size_mm=config['voxel_size_mm'],
        dilate_iters=config['dilate_iters']
    )

if __name__ == "__main__":
    main()