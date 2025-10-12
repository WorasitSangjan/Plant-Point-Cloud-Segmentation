# Plant Point Cloud Segmentation

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
[![Platform](https://img.shields.io/badge/Platform-3D%20Camera|LiDAR-red.svg)](https://raspberrypi.org)
![Agriculture](https://img.shields.io/badge/Agriculture-Precision%20Agriculture-green.svg)
![Research](https://img.shields.io/badge/Research-USDA--ARS-navy.svg)

> **Automated segmentation and volumetric analysis of individual plants from 3D point cloud scans (3D camera) using spatial clustering and voxel-based measurements.**

## Features

- **Autimatic Segmentation**: Automatically detects and separates individual plants using X-axis distribution analysis
- **Robust Cleaning**: Removes background walls, pots, and outliers using color and spatial filtering
- **Volumetric Analysis**: Multi-resolution voxel-based volume and surface area calculation
- **Visual Outputs**: Generates 3D meshes (OBJ), point clouds (PLY), and diagnostic visualizations
- **Batch Processing**: Process multiple scans automatically
- **Interactive Mode**: User-friendly prompts 

For plant phenotyping, growth analysis, and agricultural research.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Plant-Point-Cloud-Segmentation.git
cd Plant-Point-Cloud-Segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Segmentation

Segment a multi-plant point cloud into individual plants:

```bash
python segmentation.py
```

Follow the interactive prompts:
- **Mode 1**: Single file processing
- **Mode 2**: Batch processing (entire folder)

**Input**: Multi-plant PLY point cloud  
**Output**: 
- `individual_plants/plant_01.ply` through `plant_0N.ply`
- `plants_cleaned.ply` (cleaned combined point cloud)
- `segmentation_histogram.png` (visualization)

### Step 2: Volumetric Analysis

Calculate volume and surface area for each plant:

```bash
python volumetric_analysis.py
```

**Input**: Individual plant PLY files (from Step 1)  
**Output** (per plant, at 3 resolutions):
- `plant_01_coarse_voxels_10mm.obj` - 3D voxel mesh (viewable in Meshlab/Blender)
- `plant_01_medium_voxels_5mm.obj` - Medium resolution mesh
- `plant_01_fine_voxels_3mm.obj` - Fine resolution mesh
- `plant_01_*_voxels_*mm.ply` - Voxel centers as point clouds
- `plant_measurements.csv` - Summary with volume (cm³) and area (cm²)

## Example Workflow

```bash
# 1. Segment plants from the scan
python segmentation.py
> Input PLY file: my_scan.ply
> Number of plants: 7
> Output directory: individual_plants

# 2. Analyze volumes
python volumetric_analysis.py
> Input directory: individual_plants
> Output directory: results
> Base voxel size: 5
```

## Output Examples

### Segmentation Results
```
individual_plants/
├── plant_01.ply (15,234 points)
├── plant_02.ply (18,901 points)
├── plant_03.ply (14,567 points)
└── ...
```

### Volumetric Analysis Results
```
results/
├── plant_01_cleaned.ply
├── plant_01_coarse_voxels_10mm.obj
├── plant_01_medium_voxels_5mm.obj
├── plant_01_fine_voxels_3mm.obj
└── plant_measurements.csv
```

**CSV Output:**
| Plant    | Points | Coarse Vol (cm³) | Medium Vol (cm³) | Fine Vol (cm³) |
|----------|--------|------------------|------------------|----------------|
| plant_01 | 15,234 | 1,245.3          | 1,189.7          | 1,156.2        |
| plant_02 | 18,901 | 1,567.8          | 1,498.4          | 1,467.9        |

## Parameters

### Segmentation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_plants` | 7 | Expected number of plants in scan |
| `nb_neighbors` | 20 | Neighbors for outlier removal |
| `std_ratio` | 2.0 | Standard deviation threshold |

### Volumetric Analysis Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_unit` | mm | Units of input files (m/cm/mm) |
| `voxel_size_mm` | 5.0 | Base voxel size (coarse=2x, fine=0.5x) |
| `dilate_iters` | 1 | Dilation iterations to fill gaps |

## Requirements

- Python >= 3.8
- open3d >= 0.13.0
- numpy >= 1.19.0
- scipy >= 1.5.0
- matplotlib >= 3.3.0
- scikit-learn >= 0.24.0

See `requirements.txt` for a complete list.

## File Format

### Input Point Cloud Requirements

- **Format**: PLY (binary or ASCII)
- **Coordinates**: X, Y, Z in millimeters (configurable)
- **Colors**: RGB values (0-255 or 0-1)
- **Structure**: Multi-plant scene with pots visible

### Expected Coordinate System

- **X-axis**: Horizontal (plant separation axis)
- **Y-axis**: Depth (front-to-back)
- **Z-axis**: Vertical (height)

## Algorithms

### Segmentation Method

1. **Background Removal**: Depth-based filtering using white pot labels
2. **Pot Removal**: Y-coordinate analysis with color filtering
3. **Outlier Removal**: Statistical outlier detection (Open3D)
4. **Automatic Segmentation**: Histogram-based peak/valley detection in X-axis
5. **Boundary Optimization**: Automatic merging/splitting to match plant count

### Volumetric Method

1. **Cleaning**: Color filtering (green/yellow) + DBSCAN spatial clustering
2. **Voxelization**: Convert point cloud to 3D occupancy grid
3. **Dilation**: Fill small gaps using 6-neighborhood morphology
4. **Volume**: Count occupied voxels × voxel volume
5. **Surface Area**: Face counting method at voxel boundaries

## Troubleshooting

### Issue: "No PLY files found"
**Solution**: Ensure the input directory contains `.ply` files and the path is correct.

### Issue: "Zero volume results"
**Solution**: 
- Check input units match your data (mm/cm/m)
- Verify PLY files contain valid point data
- Try disabling color filtering if plants aren't green

### Issue: "Segmentation creates wrong number of plants"
**Solution**: 
- Adjust `num_plants` parameter
- Check if plants are clearly separated in the X-axis
- Verify background/pots are properly removed

### Issue: "Too many/few points after cleaning"
**Solution**: Adjust cleaning parameters:
- `nb_neighbors`: Higher = more aggressive cleaning
- `std_ratio`: Lower = more aggressive cleaning

## Citation

If you use this code in your research, please cite:

```bibtex
@software{plant_point_cloud_segmentation,
  title={Plant Point Cloud Segmentation},
  author={Worasit Sangjan},
  year={2024},
  url={https://github.com/WorasitSangjan/Plant-Point-Cloud-Segmentation}
}
```

## License

This project is open source under the [MIT License](LICENSE). See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact [worasitsangjan.ws@gmail.com]

## Acknowledgments

- Built with [Open3D](http://www.open3d.org/)
- Inspired by plant phenotyping research in precision agriculture
