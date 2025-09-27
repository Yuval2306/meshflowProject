#!/usr/bin/env python3
"""
MeshFlow Video Stabilization Implementation
Based on the 2016 paper: "MeshFlow: Minimum Latency Online Video Stabilization" by Liu et al.

This implementation follows the academic project requirements for Tel-Hai Academic College
Computer Vision Course (199615) - Video Stabilization Project

Authors: Mor Bone (318408465), Yuval Buker (211854062)
"""

import cv2
import numpy as np
import os
import sys
import time
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import argparse
from pathlib import Path


class MeshFlowStabilizer:
    """
    MeshFlow Video Stabilization Algorithm Implementation

    Based on the paper: "MeshFlow: Minimum Latency Online Video Stabilization"
    by Shuaicheng Liu, Ping Tan, Lu Yuan, Jian Sun, Bing Zeng (ECCV 2016)
    """

    def __init__(self,
                 mesh_rows: int = 16,
                 mesh_cols: int = 16,
                 motion_radius: int = 300,
                 temporal_smooth_radius: int = 10,
                 optimization_iterations: int = 100,
                 visualize: bool = False):
        """
        Initialize MeshFlow stabilizer

        Args:
            mesh_rows: Number of mesh rows (default: 16)
            mesh_cols: Number of mesh columns (default: 16)
            motion_radius: Motion propagation radius (default: 300)
            temporal_smooth_radius: Temporal smoothing radius (default: 10)
            optimization_iterations: Number of optimization iterations (default: 100)
            visualize: Whether to show visualization (default: False)
        """
        self.mesh_rows = mesh_rows
        self.mesh_cols = mesh_cols
        self.motion_radius = motion_radius
        self.temporal_smooth_radius = temporal_smooth_radius
        self.optimization_iterations = optimization_iterations
        self.visualize = visualize

        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=1000,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        # Performance metrics
        self.performance_metrics = {
            'processing_times': [],
            'stabilization_quality': [],
            'motion_vectors': []
        }

    def create_mesh(self, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a regular mesh grid over the image

        Args:
            height: Image height
            width: Image width

        Returns:
            Tuple of (mesh_x, mesh_y) coordinates
        """
        # Calculate mesh spacing
        mesh_h = height // self.mesh_rows
        mesh_w = width // self.mesh_cols

        # Create mesh coordinates
        y_coords = np.arange(0, height + 1, mesh_h)
        x_coords = np.arange(0, width + 1, mesh_w)

        mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)

        return mesh_x, mesh_y

    def detect_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect corner features in frame

        Args:
            frame: Input frame (grayscale)

        Returns:
            Array of detected feature points
        """
        corners = cv2.goodFeaturesToTrack(frame, **self.feature_params)
        return corners if corners is not None else np.array([])

    def track_features(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                       prev_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track features between consecutive frames using Lucas-Kanade

        Args:
            prev_frame: Previous frame (grayscale)
            curr_frame: Current frame (grayscale)
            prev_features: Features from previous frame

        Returns:
            Tuple of (tracked_features, status)
        """
        if len(prev_features) == 0:
            return np.array([]), np.array([])

        next_features, status, error = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, prev_features, None, **self.lk_params)

        # Filter good features
        good_mask = status.flatten() == 1
        good_features = next_features[good_mask]
        good_prev = prev_features[good_mask]

        return good_features, good_prev

    def motion_propagation(self, features_prev: np.ndarray, features_curr: np.ndarray,
                           mesh_x: np.ndarray, mesh_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate motion from features to mesh vertices using median filtering

        Args:
            features_prev: Previous frame features
            features_curr: Current frame features
            mesh_x: Mesh x coordinates
            mesh_y: Mesh y coordinates

        Returns:
            Tuple of (motion_x, motion_y) at mesh vertices
        """
        if len(features_prev) == 0 or len(features_curr) == 0:
            return np.zeros_like(mesh_x), np.zeros_like(mesh_y)

        # Handle different feature array formats
        if features_prev.ndim == 3:
            # Format: (n, 1, 2) - squeeze out middle dimension
            features_prev_2d = features_prev.squeeze(1)
            features_curr_2d = features_curr.squeeze(1)
        else:
            # Format: (n, 2) - already in correct format
            features_prev_2d = features_prev
            features_curr_2d = features_curr

        # Calculate motion vectors from features
        motion_vectors = features_curr_2d - features_prev_2d

        # Initialize motion arrays for mesh
        motion_x = np.zeros_like(mesh_x, dtype=np.float32)
        motion_y = np.zeros_like(mesh_y, dtype=np.float32)

        # For each mesh vertex, collect nearby motion vectors
        for i in range(mesh_x.shape[0]):
            for j in range(mesh_x.shape[1]):
                vertex_x = mesh_x[i, j]
                vertex_y = mesh_y[i, j]

                # Find features within radius
                distances = np.sqrt((features_prev_2d[:, 0] - vertex_x) ** 2 +
                                    (features_prev_2d[:, 1] - vertex_y) ** 2)
                nearby_indices = distances < self.motion_radius

                if np.sum(nearby_indices) > 0:
                    nearby_motions = motion_vectors[nearby_indices]
                    # Use median filtering as in the paper
                    motion_x[i, j] = np.median(nearby_motions[:, 0])
                    motion_y[i, j] = np.median(nearby_motions[:, 1])

        return motion_x, motion_y

    def smooth_vertex_profiles(self, vertex_profiles_x: List[np.ndarray],
                               vertex_profiles_y: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Smooth vertex motion profiles using PAPS (Predicted Adaptive Path Smoothing)

        Args:
            vertex_profiles_x: List of x motion profiles for each frame
            vertex_profiles_y: List of y motion profiles for each frame

        Returns:
            Tuple of smoothed (x_profiles, y_profiles)
        """
        num_frames = len(vertex_profiles_x)
        if num_frames == 0:
            return [], []

        mesh_shape = vertex_profiles_x[0].shape
        smoothed_x = []
        smoothed_y = []

        for frame_idx in range(num_frames):
            smooth_x = np.zeros_like(vertex_profiles_x[frame_idx])
            smooth_y = np.zeros_like(vertex_profiles_y[frame_idx])

            # For each vertex
            for i in range(mesh_shape[0]):
                for j in range(mesh_shape[1]):
                    # Collect temporal window around current frame
                    start_frame = max(0, frame_idx - self.temporal_smooth_radius)
                    end_frame = min(num_frames, frame_idx + self.temporal_smooth_radius + 1)

                    x_values = [vertex_profiles_x[k][i, j] for k in range(start_frame, end_frame)]
                    y_values = [vertex_profiles_y[k][i, j] for k in range(start_frame, end_frame)]

                    # Apply smoothing (simple moving average for this implementation)
                    smooth_x[i, j] = np.mean(x_values)
                    smooth_y[i, j] = np.mean(y_values)

            smoothed_x.append(smooth_x)
            smoothed_y.append(smooth_y)

        return smoothed_x, smoothed_y

    def warp_frame(self, frame: np.ndarray, mesh_x: np.ndarray, mesh_y: np.ndarray,
                   motion_x: np.ndarray, motion_y: np.ndarray) -> np.ndarray:
        """
        Warp frame according to mesh motion

        Args:
            frame: Input frame
            mesh_x: Mesh x coordinates
            mesh_y: Mesh y coordinates
            motion_x: X motion at mesh vertices
            motion_y: Y motion at mesh vertices

        Returns:
            Warped frame
        """
        height, width = frame.shape[:2]

        # Create new mesh positions
        new_mesh_x = mesh_x + motion_x
        new_mesh_y = mesh_y + motion_y

        # Create dense mapping using interpolation
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)

        # For each pixel, find which mesh cell it belongs to and interpolate
        mesh_h = height // self.mesh_rows
        mesh_w = width // self.mesh_cols

        for y in range(height):
            for x in range(width):
                # Find mesh cell
                cell_i = min(y // mesh_h, self.mesh_rows - 1)
                cell_j = min(x // mesh_w, self.mesh_cols - 1)

                # Bilinear interpolation within mesh cell
                if cell_i < mesh_x.shape[0] - 1 and cell_j < mesh_x.shape[1] - 1:
                    # Get four corners of mesh cell
                    x1, y1 = mesh_x[cell_i, cell_j], mesh_y[cell_i, cell_j]
                    x2, y2 = mesh_x[cell_i, cell_j + 1], mesh_y[cell_i, cell_j + 1]
                    x3, y3 = mesh_x[cell_i + 1, cell_j], mesh_y[cell_i + 1, cell_j]
                    x4, y4 = mesh_x[cell_i + 1, cell_j + 1], mesh_y[cell_i + 1, cell_j + 1]

                    # Get motion at four corners
                    mx1, my1 = motion_x[cell_i, cell_j], motion_y[cell_i, cell_j]
                    mx2, my2 = motion_x[cell_i, cell_j + 1], motion_y[cell_i, cell_j + 1]
                    mx3, my3 = motion_x[cell_i + 1, cell_j], motion_y[cell_i + 1, cell_j]
                    mx4, my4 = motion_x[cell_i + 1, cell_j + 1], motion_y[cell_i + 1, cell_j + 1]

                    # Bilinear interpolation
                    local_x = (x - x1) / max(1, x2 - x1)
                    local_y = (y - y1) / max(1, y3 - y1)

                    interp_mx = (1 - local_x) * (1 - local_y) * mx1 + \
                                local_x * (1 - local_y) * mx2 + \
                                (1 - local_x) * local_y * mx3 + \
                                local_x * local_y * mx4

                    interp_my = (1 - local_x) * (1 - local_y) * my1 + \
                                local_x * (1 - local_y) * my2 + \
                                (1 - local_x) * local_y * my3 + \
                                local_x * local_y * my4

                    map_x[y, x] = x - interp_mx  # Inverse mapping
                    map_y[y, x] = y - interp_my
                else:
                    map_x[y, x] = x
                    map_y[y, x] = y

        # Apply remapping
        warped_frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        return warped_frame

    def calculate_metrics(self, original_frame: np.ndarray, stabilized_frame: np.ndarray) -> dict:
        """
        Calculate stabilization performance metrics

        Args:
            original_frame: Original unstabilized frame
            stabilized_frame: Stabilized frame

        Returns:
            Dictionary of performance metrics
        """
        # Simple metrics calculation
        # This can be extended with more sophisticated measures

        # Calculate frame difference (stability metric)
        if len(original_frame.shape) == 3:
            orig_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            stab_gray = cv2.cvtColor(stabilized_frame, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original_frame
            stab_gray = stabilized_frame

        # Calculate mean squared difference
        mse = np.mean((orig_gray.astype(np.float32) - stab_gray.astype(np.float32)) ** 2)

        # Calculate PSNR
        psnr = 20 * np.log10(255.0 / np.sqrt(mse + 1e-8))

        return {
            'mse': mse,
            'psnr': psnr,
            'stability_score': 1.0 / (1.0 + mse / 1000.0)  # Normalized stability
        }

    def stabilize_video(self, input_path: str, output_path: str) -> dict:
        """
        Main video stabilization function

        Args:
            input_path: Path to input video
            output_path: Path to output stabilized video

        Returns:
            Dictionary of performance results
        """
        print(f"Starting MeshFlow video stabilization...")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Create mesh
        mesh_x, mesh_y = self.create_mesh(height, width)
        print(f"Created mesh: {mesh_x.shape}")

        # Storage for vertex profiles
        vertex_profiles_x = []
        vertex_profiles_y = []
        original_frames = []

        prev_gray = None
        prev_features = None
        frame_count = 0
        start_time = time.time()

        # First pass: collect all motion data
        print("First pass: Collecting motion data...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            original_frames.append(frame.copy())

            if prev_gray is not None and prev_features is not None:
                # Track features
                curr_features, prev_features_tracked = self.track_features(
                    prev_gray, gray, prev_features)

                # Motion propagation to mesh
                motion_x, motion_y = self.motion_propagation(
                    prev_features_tracked, curr_features, mesh_x, mesh_y)

                vertex_profiles_x.append(motion_x)
                vertex_profiles_y.append(motion_y)

                # Store motion vectors for analysis
                if len(curr_features) > 0 and len(prev_features_tracked) > 0:
                    # Handle different array formats
                    if curr_features.ndim == 3:
                        curr_2d = curr_features.squeeze(1)
                        prev_2d = prev_features_tracked.squeeze(1)
                    else:
                        curr_2d = curr_features
                        prev_2d = prev_features_tracked

                    motion_vectors = curr_2d - prev_2d
                    avg_motion = np.mean(np.sqrt(motion_vectors[:, 0] ** 2 +
                                                 motion_vectors[:, 1] ** 2))
                    self.performance_metrics['motion_vectors'].append(avg_motion)
            else:
                # First frame - no motion
                vertex_profiles_x.append(np.zeros_like(mesh_x))
                vertex_profiles_y.append(np.zeros_like(mesh_y))

            # Detect features for next iteration
            prev_features = self.detect_features(gray)
            prev_gray = gray
            frame_count += 1

            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        print(f"Collected motion data for {len(vertex_profiles_x)} frames")

        # Smooth vertex profiles using PAPS
        print("Smoothing vertex profiles...")
        smoothed_x, smoothed_y = self.smooth_vertex_profiles(vertex_profiles_x, vertex_profiles_y)

        # Second pass: apply stabilization
        print("Second pass: Applying stabilization...")
        for i, frame in enumerate(original_frames):
            frame_start_time = time.time()

            if i < len(smoothed_x):
                # Apply warping
                stabilized_frame = self.warp_frame(frame, mesh_x, mesh_y,
                                                   smoothed_x[i], smoothed_y[i])
            else:
                stabilized_frame = frame

            # Write frame
            out.write(stabilized_frame)

            # Calculate metrics
            if i > 0:  # Skip first frame
                metrics = self.calculate_metrics(frame, stabilized_frame)
                self.performance_metrics['stabilization_quality'].append(
                    metrics['stability_score'])

            frame_time = time.time() - frame_start_time
            self.performance_metrics['processing_times'].append(frame_time)

            if (i + 1) % 10 == 0:
                print(f"Stabilized {i + 1}/{len(original_frames)} frames")

        # Cleanup
        cap.release()
        out.release()

        total_time = time.time() - start_time
        avg_fps = len(original_frames) / total_time

        # Compile results
        results = {
            'total_frames': len(original_frames),
            'processing_time': total_time,
            'average_fps': avg_fps,
            'average_frame_time': np.mean(self.performance_metrics['processing_times']),
            'average_stability_score': np.mean(self.performance_metrics['stabilization_quality']) if
            self.performance_metrics['stabilization_quality'] else 0,
            'average_motion_magnitude': np.mean(self.performance_metrics['motion_vectors']) if self.performance_metrics[
                'motion_vectors'] else 0
        }

        print(f"\nStabilization completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average stability score: {results['average_stability_score']:.3f}")

        return results


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='MeshFlow Video Stabilization')
    parser.add_argument('input', help='Input video path')
    parser.add_argument('output', help='Output video path')
    parser.add_argument('--mesh-rows', type=int, default=16, help='Number of mesh rows')
    parser.add_argument('--mesh-cols', type=int, default=16, help='Number of mesh columns')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize stabilizer
    stabilizer = MeshFlowStabilizer(
        mesh_rows=args.mesh_rows,
        mesh_cols=args.mesh_cols,
        visualize=args.visualize
    )

    try:
        # Stabilize video
        results = stabilizer.stabilize_video(args.input, args.output)

        # Print results
        print("\n" + "=" * 50)
        print("STABILIZATION RESULTS")
        print("=" * 50)
        for key, value in results.items():
            print(f"{key}: {value}")

        return 0

    except Exception as e:
        print(f"Error during stabilization: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())