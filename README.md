# MeshFlow Video Stabilization - Final Project

**Tel-Hai Academic College - Computer Science Department**  
**Course: Computer Vision (199615)**  
**Project Topic: Video Stabilization**

**Authors:**
- Mor Bone (318408465)
- Yuval Buker (211854062)

**Instructor:** Uri Brit  
**Submission Date:** October 2025

---

## Project Overview

This project implements the MeshFlow video stabilization algorithm based on the paper:
**"MeshFlow: Minimum Latency Online Video Stabilization"** by Liu et al. (ECCV 2016)

The project includes:
- Complete Python implementation of MeshFlow algorithm
- Comprehensive performance evaluation and limitations analysis
- Original research on real-time optimization
- Detailed analysis of stabilization quality and computational efficiency

---

## Project Files

```
meshflow_project/
├── meshflow_stabilizer.py          # Basic MeshFlow implementation
├── demo.py                         # Simple demonstration
├── test_and_evaluation.py          # Comprehensive testing system
├── real_time_meshflow.py           # Real-time optimized version (research)
├── requirements.txt                # Dependencies list
├── README.md                       # This file
├── .gitignore                      # Git ignore file
└── .idea/                          # IDE settings (ignored by git)
```

---

## Installation and Setup

### 1. Download Project
```bash
# Download all files to a new directory
mkdir meshflow_project
cd meshflow_project
```

### 2. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Or manually:
pip install opencv-python numpy scipy matplotlib pandas tqdm
```

### 3. Verify Installation
```bash
# Quick test that everything works
python -c "import cv2, numpy, scipy, matplotlib; print('Setup OK!')"
```

---

## Running the Project

### Step 1: Basic Demonstration
```bash
python demo.py
```
**What happens:**
- Creates sample shaky video
- Runs MeshFlow stabilization
- Creates stabilized video
- Generates side-by-side comparison

**Expected runtime:** 5-10 minutes  
**Output:** `results/stabilized_video.mp4`, `results/comparison_video.mp4`

### Step 2: Comprehensive Evaluation (Implementation Tasks)
```bash
python test_and_evaluation.py
```
**What happens:**
- Creates 5 different test video types
- Tests multiple parameter configurations
- Analyzes research questions
- Identifies algorithm limitations
- Generates plots and reports

**Expected runtime:** 20-40 minutes  
**Output:** `evaluation_results/` with graphs, data, and videos

### Step 3: Research - Real-time Optimization
```bash
python real_time_meshflow.py
```
**What happens:**
- Compares original vs optimized version
- Tests speed improvements
- Analyzes quality vs speed trade-offs

**Expected runtime:** 10-15 minutes  
**Output:** Comparison files and performance data

### Step 4: Custom Video Stabilization
```bash
# With your own video
python meshflow_stabilizer.py input_video.mp4 output_video.mp4

# With custom parameters
python meshflow_stabilizer.py input.mp4 output.mp4 --mesh-rows 20 --mesh-cols 20
```

---

## Expected Results

### Basic Implementation
- **Processing Speed:** 0.6-1.0 FPS
- **Stabilization Quality:** 70-90% improvement in shakiness
- **Processing Time:** 3-5 minutes for 10-second video

### Optimized Version (Research)
- **Processing Speed:** 8-15 FPS
- **Speed Improvement:** 10-25x faster
- **Quality Retention:** 80-95% of original quality maintained

### Success Metrics
- **Stability Score:** 0.7-0.9 (scale 0-1)
- **Motion Reduction:** 60-90% reduction in unwanted motion
- **Processing Efficiency:** Significant time/quality ratio improvement

---

## Troubleshooting

### Issue: NumPy compatibility error
```bash
pip install "numpy<2.0"
# or
pip install numpy==1.24.3
```

### Issue: Slow processing
- Reduce mesh size: `mesh_rows=8, mesh_cols=8`
- Reduce video resolution
- Use optimized version: `real_time_meshflow.py`

### Issue: Video won't open
- Ensure supported format (MP4 preferred)
- Check file path
- Try converting to different format

### Issue: Memory issues
- Work with smaller video
- Close other applications
- Reduce mesh size

---

## Results Structure

### `results/` Directory
```
results/
├── stabilized_video.mp4           # Stabilized video from demo
├── comparison_video.mp4           # Side-by-side comparison
└── demo_videos/                   # Original sample video
```

### `evaluation_results/` Directory
```
evaluation_results/
├── plots/                         # Graphs and visualizations
│   ├── performance_comparison.png
│   ├── stability_metrics.png
│   └── processing_times.png
├── videos/                        # Test videos and results
│   ├── test_linear_shake.mp4
│   ├── test_circular_motion.mp4
│   └── comparison videos
└── data/                          # Detailed data
    ├── comprehensive_report.json
    ├── research_questions_analysis.json
    └── limitations_analysis.json
```

---

## Advanced Parameters

### Basic MeshFlow
- `mesh_rows/cols`: Mesh resolution (8-32)
- `temporal_smooth_radius`: Temporal smoothing radius (5-20)
- `motion_radius`: Propagation radius (150-500)

### Optimized Version
- `base_mesh_size`: Base mesh size (6-12)
- `max_mesh_size`: Maximum mesh size (12-24)
- `temporal_window`: Reduced temporal window (2-5)
- `enable_parallel`: Parallel processing (True/False)

---

## Performance Summary

### Implementation Tasks Results
Our comprehensive evaluation shows:

| Video Type | Stability Score | Motion Reduction | Processing Speed |
|-----------|----------------|------------------|------------------|
| Linear shake | 0.910 | 85% | 0.65 FPS |
| Circular motion | 0.838 | 78% | 0.58 FPS |
| Random jitter | 0.823 | 72% | 0.59 FPS |
| Zoom shake | 0.984 | 92% | 0.64 FPS |
| Complex motion | 0.876 | 81% | 0.61 FPS |

### Research Results
Real-time optimization achieved:
- **15.9x speed improvement**: From 0.65 to 10.43 FPS
- **Quality retention**: ~85% of original stabilization quality
- **Memory efficiency**: 60% reduction in memory usage
- **Latency reduction**: From 10 to 3 frames delay

---

## Academic Context

### Course Requirements Addressed
This project fulfills all Implementation Tasks requirements:

✅ **Algorithm Implementation**: Complete MeshFlow implementation  
✅ **Performance Testing**: Comprehensive evaluation with multiple test scenarios  
✅ **Research Questions**: Analysis of 4 key research questions  
✅ **Limitations Analysis**: Identification of failure modes and bottlenecks  
✅ **Original Research**: Real-time optimization with significant improvements  

### Research Contributions
1. **Real-time adaptation**: 15x speed improvement while maintaining quality
2. **Comprehensive evaluation**: Systematic analysis of algorithm limitations
3. **Practical insights**: Trade-offs between quality, speed, and resources
4. **Future directions**: Concrete suggestions for further improvements

---

## Future Extensions

### Technical Improvements
- GPU acceleration with CUDA
- Deep learning integration
- Real-time streaming support

### Algorithm Enhancements
- Adaptive mesh with ML
- Multi-scale temporal filtering
- Robust motion estimation

### Applications
- Live streaming stabilization
- Mobile device optimization
- VR/AR content stabilization

---

## License and Credits

- Based on the open paper: Liu et al., "MeshFlow: Minimum Latency Online Video Stabilization", ECCV 2016
- Code written entirely for this academic project
- Usage permitted for educational and research purposes
- Not for commercial use without permission

---

## Contact and Support

For technical questions or issues:
1. Check troubleshooting section above
2. Verify correct installation of all dependencies
3. Check file permissions for read/write access
4. Contact course instructor for additional guidance

