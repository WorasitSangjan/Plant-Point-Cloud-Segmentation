# Developed: Worasit Sangjan
# Date: 7 October 2025

import os
import glob
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION (Optional: Edit these defaults)
# ============================================================================

DEFAULT_INPUT = "Cloud_20240820_060213_364090.ply"
DEFAULT_NUM_PLANTS = 7
DEFAULT_OUTPUT_DIR = "individual_plants"
DEFAULT_NB_NEIGHBORS = 20
DEFAULT_STD_RATIO = 2.0

# ============================================================================
# PART 1: CLEANING 
# ============================================================================

def load_point_cloud(filename):
    """Load point cloud from PLY file."""
    print("="*70)
    print("STEP 1: REMOVE BACK WALL")
    print("="*70)
    
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    
    print(f"\nTotal points: {len(points):,}")
    print(f"Z range: [{points[:, 2].min():.1f}, {points[:, 2].max():.1f}]mm")
    
    return points, colors

def remove_back_wall(points, colors, depth_threshold=200):
    """Remove back wall using white pot ID labels as reference."""
    brightness = colors.mean(axis=1)
    R, G, B = colors[:, 0], colors[:, 1], colors[:, 2]
    color_variance = np.std([R, G, B], axis=0)
    
    # Detect white pot ID labels
    white_mask = (brightness > 200) & (color_variance < 30)
    
    if np.sum(white_mask) < 100:
        id_z = np.percentile(points[:, 2], 5)
    else:
        white_points = points[white_mask]
        z_values = white_points[:, 2]
        id_z = np.percentile(z_values, 5)
    
    print(f"White ID reference: Z = {id_z:.1f}mm")
    print(f"Keeping: Z < {id_z + depth_threshold:.1f}mm")
    
    # Keep within depth threshold
    depth_mask = points[:, 2] < (id_z + depth_threshold)
    
    # Remove dark backdrop
    dark_mask = brightness < 75
    
    # Final mask
    wall_removed_mask = depth_mask & (~dark_mask)
    
    kept_pct = 100 * np.sum(wall_removed_mask) / len(points)
    print(f"After removing back wall: {np.sum(wall_removed_mask):,} points ({kept_pct:.1f}%)")
    
    return points[wall_removed_mask], colors[wall_removed_mask]

def remove_pots(points, colors):
    """Remove pot bodies using Y coordinate (horizontal position)."""
    print("\n" + "="*70)
    print("STEP 2: REMOVE POTS")
    print("="*70)
    
    R, G, B = colors[:, 0], colors[:, 1], colors[:, 2]
    
    # Identify green (plant) vs non-green (pot) points
    green_mask = (G > 110) & (G > R * 1.1) & (G > B * 1.15)
    
    # Analyze Y coordinate distribution
    print("\nAnalyzing Y coordinate distribution:")
    y_values = points[:, 1]
    print(f"  Y range: [{y_values.min():.1f}, {y_values.max():.1f}]mm")
    
    green_y_avg = points[green_mask, 1].mean() if np.sum(green_mask) > 0 else 0
    non_green_y_avg = points[~green_mask, 1].mean() if np.sum(~green_mask) > 0 else 0
    
    print(f"  Green (plants) average Y: {green_y_avg:.1f}mm")
    print(f"  Non-green (pots) average Y: {non_green_y_avg:.1f}mm")
    
    # Determine pot position
    pot_at_low_y = green_y_avg > non_green_y_avg
    direction = "LOWER" if pot_at_low_y else "HIGHER"
    print(f"  → Pots at {direction} Y, plants at opposite side")
    
    # Find pot region in Y
    non_green_y = points[~green_mask, 1]
    
    if len(non_green_y) > 100:
        pot_y_threshold = np.percentile(non_green_y, 66.25)
        print(f"\nPot Y threshold: {pot_y_threshold:.1f}mm")
        
        if pot_at_low_y:
            plant_mask = y_values > pot_y_threshold
            print(f"Keeping Y > {pot_y_threshold:.1f}mm (plants)")
        else:
            plant_mask = y_values < pot_y_threshold
            print(f"Keeping Y < {pot_y_threshold:.1f}mm (plants)")
    else:
        print("Cannot determine pot Y location, skipping pot removal")
        plant_mask = np.ones(len(points), dtype=bool)
    
    removed_pots = np.sum(~plant_mask)
    kept_pct = 100 * np.sum(plant_mask) / len(points)
    print(f"\nRemoved pot points: {removed_pots:,}")
    print(f"Remaining (plants): {np.sum(plant_mask):,} points ({kept_pct:.1f}%)")
    
    return points[plant_mask], colors[plant_mask]

def remove_outliers(points, colors, nb_neighbors=20, std_ratio=2.0):
    """Remove statistical outliers and isolated points."""
    print("\n" + "="*70)
    print("STEP 3: CLEAN AND REMOVE OUTLIERS")
    print("="*70)
    
    # Create point cloud for outlier removal
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    print(f"Before cleaning: {len(points):,} points")
    
    # Statistical outlier removal
    print(f"\nRemoving statistical outliers (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})...")
    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, 
        std_ratio=std_ratio
    )
    
    removed = len(points) - len(pcd_clean.points)
    print(f"Removed {removed:,} outliers")
    print(f"After cleaning: {len(pcd_clean.points):,} points")
    
    # Convert back to numpy
    points_clean = np.asarray(pcd_clean.points)
    colors_clean = np.asarray(pcd_clean.colors) * 255
    
    return points_clean, colors_clean

# ============================================================================
# PART 2: SEGMENTATION
# ============================================================================

def segment_plants_smart(points, num_plants=7, bin_width=5.0, verbose=True):
    """Smart segmentation: find natural gaps in X distribution."""
    x = points[:, 0]
    x_min, x_max = x.min(), x.max()
    
    # Create fine histogram
    num_bins = int((x_max - x_min) / bin_width)
    hist, bin_edges = np.histogram(x, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smooth histogram to reduce noise
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=2)
    
    if verbose:
        print("\n" + "="*70)
        print("STEP 4: SMART PLANT SEGMENTATION")
        print("="*70)
        print(f"X range: [{x_min:.1f}, {x_max:.1f}] mm")
        print(f"Histogram bins: {num_bins} (bin_width={bin_width:.1f}mm)")
    
    # Find peaks (plant centers)
    peak_height = np.max(hist_smooth) * 0.15
    peaks, _ = find_peaks(hist_smooth, height=peak_height, distance=5)
    
    if verbose:
        print(f"\nFound {len(peaks)} potential plant centers")
    
    # Find valleys (gaps between plants)
    valleys, _ = find_peaks(-hist_smooth, prominence=5)
    
    # Filter valleys: only keep those between peaks
    valid_valleys = []
    if len(peaks) > 1:
        for i in range(len(peaks) - 1):
            candidates = valleys[(valleys > peaks[i]) & (valleys < peaks[i+1])]
            if len(candidates) > 0:
                min_idx = np.argmin(hist_smooth[candidates])
                valid_valleys.append(candidates[min_idx])
    
    valley_positions = bin_centers[valid_valleys] if len(valid_valleys) > 0 else []
    
    if verbose:
        print(f"Found {len(valley_positions)} gaps between plants")
        if len(valley_positions) > 0:
            print(f"Gap positions (X): {valley_positions}")
    
    # Create boundaries from valleys
    boundaries = [x_min]
    boundaries.extend(sorted(valley_positions))
    boundaries.append(x_max)
    boundaries = np.array(boundaries)
    
    num_segments = len(boundaries) - 1
    
    if verbose:
        print(f"\nInitial segmentation: {num_segments} segments")
    
    # If we have too many segments, merge the smallest ones
    if num_segments > num_plants:
        if verbose:
            print(f"Too many segments ({num_segments}), merging smallest...")
        
        segment_counts = []
        for i in range(num_segments):
            mask = (x >= boundaries[i]) & (x < boundaries[i+1])
            segment_counts.append(np.sum(mask))
        
        while len(boundaries) - 1 > num_plants:
            min_idx = np.argmin(segment_counts)
            
            if min_idx < len(segment_counts) - 1:
                segment_counts[min_idx] += segment_counts[min_idx + 1]
                segment_counts.pop(min_idx + 1)
                boundaries = np.delete(boundaries, min_idx + 1)
            else:
                segment_counts[min_idx - 1] += segment_counts[min_idx]
                segment_counts.pop(min_idx)
                boundaries = np.delete(boundaries, min_idx)
    
    # If we have too few segments, split the largest ones
    elif num_segments < num_plants:
        if verbose:
            print(f"Too few segments ({num_segments}), splitting largest...")
        
        while len(boundaries) - 1 < num_plants:
            segment_widths = []
            for i in range(len(boundaries) - 1):
                mask = (x >= boundaries[i]) & (x < boundaries[i+1])
                if np.sum(mask) > 0:
                    segment_widths.append(boundaries[i+1] - boundaries[i])
                else:
                    segment_widths.append(0)
            
            max_idx = np.argmax(segment_widths)
            midpoint = (boundaries[max_idx] + boundaries[max_idx + 1]) / 2
            boundaries = np.insert(boundaries, max_idx + 1, midpoint)
    
    # Assign labels based on boundaries
    labels = np.zeros(len(x), dtype=int)
    for i in range(len(boundaries) - 1):
        mask = (x >= boundaries[i]) & (x < boundaries[i+1])
        labels[mask] = i
    
    if verbose:
        print(f"\nFinal segmentation: {num_plants} plants")
        print("\nPlant statistics:")
        for i in range(num_plants):
            mask = (labels == i)
            plant_pts = points[mask]
            if len(plant_pts) > 0:
                x_range = plant_pts[:, 0].max() - plant_pts[:, 0].min()
                height = plant_pts[:, 2].max() - plant_pts[:, 2].min()
                print(f"  Plant {i+1}: {len(plant_pts):6,} pts | "
                      f"X=[{boundaries[i]:6.1f}, {boundaries[i+1]:6.1f}] | "
                      f"Width={x_range:5.1f}mm | Height={height:5.1f}mm")
            else:
                print(f"  Plant {i+1}: EMPTY")
    
    # Return histogram data for visualization
    viz_data = {
        'bin_centers': bin_centers,
        'hist_smooth': hist_smooth,
        'peaks': peaks,
        'valleys': valid_valleys,
        'boundaries': boundaries
    }
    
    return labels, boundaries, viz_data

def visualize_segmentation(viz_data, filename="segmentation_histogram.png"):
    """Create histogram visualization showing segmentation."""
    bin_centers = viz_data['bin_centers']
    hist_smooth = viz_data['hist_smooth']
    peaks = viz_data['peaks']
    valleys = viz_data['valleys']
    boundaries = viz_data['boundaries']
    
    plt.figure(figsize=(16, 6))
    
    # Plot histogram
    plt.plot(bin_centers, hist_smooth, linewidth=2, label='Point density', color='black')
    
    # Mark peaks (plant centers)
    if len(peaks) > 0:
        plt.plot(bin_centers[peaks], hist_smooth[peaks], 'go', 
                markersize=10, label=f'Plant centers ({len(peaks)})', zorder=5)
    
    # Mark valleys (gaps)
    if len(valleys) > 0:
        plt.plot(bin_centers[valleys], hist_smooth[valleys], 'rx', 
                markersize=12, markeredgewidth=3, label=f'Gaps ({len(valleys)})', zorder=5)
    
    # Draw boundary lines
    for i, b in enumerate(boundaries[1:-1], start=1):
        plt.axvline(b, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        if i == 1:
            plt.axvline(b, color='red', linestyle='--', alpha=0.5, 
                       linewidth=1.5, label='Boundaries')
    
    # Shade regions by plant
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7', '#a29bfe', '#fd79a8']
    for i in range(len(boundaries) - 1):
        plt.axvspan(boundaries[i], boundaries[i+1], 
                   alpha=0.2, color=colors[i % len(colors)])
        # Add plant number
        mid_x = (boundaries[i] + boundaries[i+1]) / 2
        plt.text(mid_x, plt.ylim()[1] * 0.9, f'P{i+1}', 
                ha='center', fontsize=14, fontweight='bold')
    
    plt.xlabel('X coordinate (mm)', fontsize=12)
    plt.ylabel('Point count', fontsize=12)
    plt.title('Plant Segmentation - X Distribution Analysis', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved histogram: {filename}")
    plt.close()

def save_individual_plants(points, colors, labels, num_plants, output_dir="individual_plants"):
    """Save each plant as separate PLY file."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("STEP 5: SAVING INDIVIDUAL PLANTS")
    print("="*70)
    
    for i in range(num_plants):
        mask = (labels == i)
        plant_pts = points[mask]
        plant_colors = colors[mask] if colors is not None else None
        
        if len(plant_pts) == 0:
            print(f"  Plant {i+1}: Skipped (no points)")
            continue
        
        filename = os.path.join(output_dir, f"plant_{i+1:02d}.ply")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(plant_pts)
        if plant_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(plant_colors / 255.0)
        
        o3d.io.write_point_cloud(filename, pcd)
        print(f"  Saved: {filename} ({len(plant_pts):,} points)")
    
    print(f"\nAll plants saved to: {output_dir}/")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def get_user_inputs():
    """Interactively prompt user for configuration."""
    print("="*70)
    print("PLANT POINT CLOUD SEGMENTATION")
    print("="*70)
    print("\nModes:")
    print("  [1] Single file processing")
    print("  [2] Batch processing (process entire folder)")
    
    mode = input("\nSelect mode [1]: ").strip() or "1"
    
    if mode == "2":
        # Batch mode
        print("\n" + "="*70)
        print("BATCH MODE CONFIGURATION")
        print("="*70)
        
        input_dir = input(f"\nInput directory containing PLY files [{DEFAULT_INPUT}]: ").strip()
        if not input_dir:
            input_dir = os.path.dirname(DEFAULT_INPUT) or "."
        
        num_plants_input = input(f"Number of plants per file [{DEFAULT_NUM_PLANTS}]: ").strip()
        num_plants = int(num_plants_input) if num_plants_input else DEFAULT_NUM_PLANTS
        
        output_base = input(f"Base output directory [{DEFAULT_OUTPUT_DIR}_batch]: ").strip()
        if not output_base:
            output_base = f"{DEFAULT_OUTPUT_DIR}_batch"
        
        # Advanced settings
        nb_neighbors, std_ratio = get_advanced_settings()
        
        return {
            'mode': 'batch',
            'input_dir': input_dir,
            'num_plants': num_plants,
            'output_base': output_base,
            'nb_neighbors': nb_neighbors,
            'std_ratio': std_ratio
        }
    
    else:
        # Single file mode
        print("\n" + "="*70)
        print("SINGLE FILE CONFIGURATION")
        print("="*70)
        
        input_file = input(f"\nInput PLY file [{DEFAULT_INPUT}]: ").strip()
        if not input_file:
            input_file = DEFAULT_INPUT
        
        num_plants_input = input(f"Number of plants [{DEFAULT_NUM_PLANTS}]: ").strip()
        num_plants = int(num_plants_input) if num_plants_input else DEFAULT_NUM_PLANTS
        
        output_dir = input(f"Output directory [{DEFAULT_OUTPUT_DIR}]: ").strip()
        if not output_dir:
            output_dir = DEFAULT_OUTPUT_DIR
        
        # Advanced settings
        nb_neighbors, std_ratio = get_advanced_settings()
        
        return {
            'mode': 'single',
            'input_file': input_file,
            'num_plants': num_plants,
            'output_dir': output_dir,
            'nb_neighbors': nb_neighbors,
            'std_ratio': std_ratio
        }

def get_advanced_settings():
    """Prompt for advanced cleaning parameters."""
    print("\n" + "-"*70)
    advanced = input("Configure advanced cleaning parameters? (y/N): ").strip().lower()
    
    if advanced == 'y':
        nb_input = input(f"  Neighbors for outlier removal [{DEFAULT_NB_NEIGHBORS}]: ").strip()
        nb_neighbors = int(nb_input) if nb_input else DEFAULT_NB_NEIGHBORS
        
        std_input = input(f"  Std ratio for outlier removal [{DEFAULT_STD_RATIO}]: ").strip()
        std_ratio = float(std_input) if std_input else DEFAULT_STD_RATIO
    else:
        nb_neighbors = DEFAULT_NB_NEIGHBORS
        std_ratio = DEFAULT_STD_RATIO
    
    return nb_neighbors, std_ratio

def run_pipeline(input_file, num_plants, output_dir, nb_neighbors, std_ratio):
    """Execute the segmentation pipeline for a single file."""
    print("\n" + "="*70)
    print(f"PROCESSING: {input_file}")
    print("="*70)
    
    # Output paths
    basename = os.path.splitext(os.path.basename(input_file))[0]
    cleaned_output = os.path.join(output_dir, f"{basename}_cleaned.ply")
    histogram_output = os.path.join(output_dir, f"{basename}_histogram.png")
    
    # PART 1: CLEANING
    points, colors = load_point_cloud(input_file)
    original_count = len(points)
    
    points, colors = remove_back_wall(points, colors)
    points, colors = remove_pots(points, colors)
    points, colors = remove_outliers(
        points, colors, 
        nb_neighbors=nb_neighbors, 
        std_ratio=std_ratio
    )
    
    # Save cleaned result
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud(cleaned_output, pcd)
    print(f"\nSaved: {cleaned_output}")
    
    # PART 2: SEGMENTATION
    labels, boundaries, viz_data = segment_plants_smart(
        points, 
        num_plants=num_plants,
        bin_width=5.0,
        verbose=True
    )
    
    # Visualize segmentation
    visualize_segmentation(viz_data, histogram_output)
    
    # Save individual plants
    plants_output_dir = os.path.join(output_dir, f"{basename}_plants")
    save_individual_plants(points, colors, labels, num_plants, plants_output_dir)
    
    # FINAL SUMMARY
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Original points: {original_count:,}")
    print(f"Final points: {len(points):,} ({100*len(points)/original_count:.1f}%)")
    print("\nOutput files:")
    print(f"  - {cleaned_output}")
    print(f"  - {plants_output_dir}/plant_01.ply through plant_{num_plants:02d}.ply")
    print(f"  - {histogram_output}")
    
    return True

def run_batch_processing(input_dir, num_plants, output_base, nb_neighbors, std_ratio):
    """Process all PLY files in a directory."""
    # Find all PLY files
    ply_files = glob.glob(os.path.join(input_dir, "*.ply"))
    
    if not ply_files:
        print(f"\nERROR: No PLY files found in {input_dir}")
        return
    
    print("\n" + "="*70)
    print(f"BATCH PROCESSING: {len(ply_files)} files")
    print("="*70)
    for i, f in enumerate(ply_files, 1):
        print(f"  [{i}] {os.path.basename(f)}")
    print("="*70)
    
    # Process each file
    success_count = 0
    failed_files = []
    
    for i, ply_file in enumerate(ply_files, 1):
        basename = os.path.splitext(os.path.basename(ply_file))[0]
        file_output_dir = os.path.join(output_base, basename)
        os.makedirs(file_output_dir, exist_ok=True)
        
        print(f"\n\n{'='*70}")
        print(f"FILE {i}/{len(ply_files)}: {basename}")
        print("="*70)
        
        try:
            run_pipeline(
                input_file=ply_file,
                num_plants=num_plants,
                output_dir=file_output_dir,
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            success_count += 1
        except Exception as e:
            print(f"\nERROR processing {basename}: {e}")
            failed_files.append(basename)
    
    # Final batch summary
    print("\n\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Successfully processed: {success_count}/{len(ply_files)} files")
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    print(f"\nAll results saved to: {output_base}/")
    print("="*70)

def main():
    """Main entry point - interactive mode with single/batch options."""
    config = get_user_inputs()
    
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    
    if config['mode'] == 'batch':
        print(f"Mode:          Batch processing")
        print(f"Input dir:     {config['input_dir']}")
        print(f"Plants/file:   {config['num_plants']}")
        print(f"Output base:   {config['output_base']}")
        print(f"Outlier params: nb_neighbors={config['nb_neighbors']}, std_ratio={config['std_ratio']}")
        print("="*70)
        
        input("\nPress Enter to start batch processing...")
        
        run_batch_processing(
            input_dir=config['input_dir'],
            num_plants=config['num_plants'],
            output_base=config['output_base'],
            nb_neighbors=config['nb_neighbors'],
            std_ratio=config['std_ratio']
        )
    else:
        print(f"Mode:          Single file")
        print(f"Input file:    {config['input_file']}")
        print(f"Plants:        {config['num_plants']}")
        print(f"Output dir:    {config['output_dir']}")
        print(f"Outlier params: nb_neighbors={config['nb_neighbors']}, std_ratio={config['std_ratio']}")
        print("="*70)
        
        input("\nPress Enter to start processing...")
        
        os.makedirs(config['output_dir'], exist_ok=True)
        run_pipeline(
            input_file=config['input_file'],
            num_plants=config['num_plants'],
            output_dir=config['output_dir'],
            nb_neighbors=config['nb_neighbors'],
            std_ratio=config['std_ratio']
        )

if __name__ == "__main__":
    main()