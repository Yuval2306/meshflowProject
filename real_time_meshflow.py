#!/usr/bin/env python3
"""
Real-time MeshFlow Optimization - Research Task
Optimized version of MeshFlow for real-time video stabilization

Research Question: Can we achieve 15+ FPS processing while maintaining stabilization quality?

Optimizations implemented:
1. Adaptive mesh resolution based on motion complexity
2. Parallel processing of mesh operations
3. Reduced temporal smoothing window for lower latency
4. Optimized feature detection and tracking
5. Streaming processing with limited memory usage

Authors: Mor Bone (318408465), Yuval Buker (211854062)
Tel-Hai Academic College - Computer Vision Course (199615)
"""

import cv2
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from collections import deque
from typing import Tuple, List, Optional, Dict
import threading


class RealTimeMeshFlowStabilizer:
    """
    Optimized MeshFlow implementation for real-time processing

    Key optimizations:
    - Adaptive mesh resolution
    - Parallel processing
    - Reduced latency
    - Memory efficient streaming
    """

    def __init__(self,
                 base_mesh_size: int = 8,
                 max_mesh_size: int = 16,
                 temporal_window: int = 3,  # Reduced from 10
                 motion_threshold: float = 10.0,
                 enable_parallel: bool = True):
        """
        Initialize optimized stabilizer

        Args:
            base_mesh_size: Base mesh resolution (smaller = faster)
            max_mesh_size: Maximum mesh resolution for complex scenes
            temporal_window: Reduced temporal smoothing window
            motion_threshold: Threshold for adaptive mesh switching
            enable_parallel: Enable parallel processing
        """
        self.base_mesh_size = base_mesh_size
        self.max_mesh_size = max_mesh_size
        self.temporal_window = temporal_window
        self.motion_threshold = motion_threshold
        self.enable_parallel = enable_parallel

        # Current mesh resolution (adaptive)
        self.current_mesh_rows = base_mesh_size
        self.current_mesh_cols = base_mesh_size

        # Optimized feature detection parameters
        self.feature_params = dict(
            maxCorners=300,  # Reduced from 1000
            qualityLevel=0.01,  # Relaxed for speed
            minDistance=10,  # Increased for fewer features
            blockSize=3,  # Reduced block size
            useHarrisDetector=True,  # Harris is faster than Shi-Tomasi
            k=0.04
        )

        # Optimized optical flow parameters
        self.lk_params = dict(
            winSize=(10, 10),  # Reduced from (15,15)
            maxLevel=1,  # Reduced from 2
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.05)
        )

        # Streaming buffers (limited size for memory efficiency)
        self.vertex_buffer_x = deque(maxlen=temporal_window)
        self.vertex_buffer_y = deque(maxlen=temporal_window)
        self.motion_history = deque(maxlen=5)  # For adaptive mesh decision

        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.processing_stats = {
            'mesh_adaptations': 0,
            'parallel_ops': 0,
            'avg_features': 0
        }

        # Thread pool for parallel operations
        if enable_parallel:
            self.thread_pool = ThreadPoolExecutor(max_workers=min(4, mp.cpu_count()))
        else:
            self.thread_pool = None

    def adapt_mesh_resolution(self, motion_magnitude: float) -> None:
        """
        Dynamically adjust mesh resolution based on motion complexity

        Args:
            motion_magnitude: Average motion magnitude in current frame
        """
        if motion_magnitude > self.motion_threshold:
            # High motion - increase resolution for better quality
            new_size = min(self.max_mesh_size, self.current_mesh_rows + 2)
            if new_size != self.current_mesh_rows:
                self.current_mesh_rows = new_size
                self.current_mesh_cols = new_size
                self.processing_stats['mesh_adaptations'] += 1
        else:
            # Low motion - decrease resolution for speed
            new_size = max(self.base_mesh_size, self.current_mesh_rows - 1)
            if new_size != self.current_mesh_rows:
                self.current_mesh_rows = new_size
                self.current_mesh_cols = new_size

    def create_adaptive_mesh(self, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create mesh with current adaptive resolution"""
        mesh_h = height // self.current_mesh_rows
        mesh_w = width // self.current_mesh_cols

        y_coords = np.arange(0, height + 1, mesh_h)
        x_coords = np.arange(0, width + 1, mesh_w)

        mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
        return mesh_x, mesh_y

    def optimized_feature_detection(self, frame: np.ndarray) -> np.ndarray:
        """
        Optimized feature detection with ROI focus

        Args:
            frame: Input grayscale frame

        Returns:
            Detected feature points
        """
        # Focus on center region for most important features
        h, w = frame.shape
        roi_margin = min(50, h // 8, w // 8)
        roi = frame[roi_margin:-roi_margin, roi_margin:-roi_margin]

        # Detect features in ROI
        corners = cv2.goodFeaturesToTrack(roi, **self.feature_params)

        if corners is not None:
            # Adjust coordinates back to full frame
            corners[:, 0, 0] += roi_margin
            corners[:, 0, 1] += roi_margin

            # Add some features from full frame for completeness
            full_corners = cv2.goodFeaturesToTrack(frame, maxCorners=100,
                                                   qualityLevel=0.05, minDistance=20, blockSize=3)
            if full_corners is not None:
                corners = np.vstack([corners, full_corners])

        return corners if corners is not None else np.array([])

    def parallel_motion_propagation(self, features_prev: np.ndarray,
                                    features_curr: np.ndarray,
                                    mesh_x: np.ndarray,
                                    mesh_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parallel implementation of motion propagation

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

        # Handle array format
        if features_prev.ndim == 3:
            features_prev_2d = features_prev.squeeze(1)
            features_curr_2d = features_curr.squeeze(1)
        else:
            features_prev_2d = features_prev
            features_curr_2d = features_curr

        motion_vectors = features_curr_2d - features_prev_2d

        motion_x = np.zeros_like(mesh_x, dtype=np.float32)
        motion_y = np.zeros_like(mesh_y, dtype=np.float32)

        def process_mesh_region(row_range):
            """Process a region of mesh vertices"""
            for i in row_range:
                for j in range(mesh_x.shape[1]):
                    vertex_x = mesh_x[i, j]
                    vertex_y = mesh_y[i, j]

                    # Optimized distance calculation using broadcasting
                    distances = np.sqrt((features_prev_2d[:, 0] - vertex_x) ** 2 +
                                        (features_prev_2d[:, 1] - vertex_y) ** 2)
                    nearby_indices = distances < 150  # Reduced radius for speed

                    if np.sum(nearby_indices) > 0:
                        nearby_motions = motion_vectors[nearby_indices]
                        motion_x[i, j] = np.median(nearby_motions[:, 0])
                        motion_y[i, j] = np.median(nearby_motions[:, 1])

        if self.enable_parallel and self.thread_pool and mesh_x.shape[0] > 4:
            # Split mesh rows across threads
            num_threads = min(4, mesh_x.shape[0])
            rows_per_thread = mesh_x.shape[0] // num_threads

            futures = []
            for thread_idx in range(num_threads):
                start_row = thread_idx * rows_per_thread
                end_row = (thread_idx + 1) * rows_per_thread if thread_idx < num_threads - 1 else mesh_x.shape[0]
                row_range = range(start_row, end_row)

                future = self.thread_pool.submit(process_mesh_region, row_range)
                futures.append(future)

            # Wait for all threads to complete
            for future in futures:
                future.result()

            self.processing_stats['parallel_ops'] += 1
        else:
            # Single-threaded processing
            process_mesh_region(range(mesh_x.shape[0]))

        return motion_x, motion_y

    def fast_temporal_smoothing(self, motion_x: np.ndarray,
                                motion_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast temporal smoothing using limited history

        Args:
            motion_x: Current frame x motion
            motion_y: Current frame y motion

        Returns:
            Smoothed motion vectors
        """
        # Add current motion to buffers
        self.vertex_buffer_x.append(motion_x.copy())
        self.vertex_buffer_y.append(motion_y.copy())

        if len(self.vertex_buffer_x) < 2:
            return motion_x, motion_y

        # Simple weighted average with more weight on recent frames
        weights = np.array([0.1, 0.3, 0.6])[:len(self.vertex_buffer_x)]
        weights = weights / np.sum(weights)

        smoothed_x = np.zeros_like(motion_x)
        smoothed_y = np.zeros_like(motion_y)

        for i, w in enumerate(weights):
            smoothed_x += w * self.vertex_buffer_x[i]
            smoothed_y += w * self.vertex_buffer_y[i]

        return smoothed_x, smoothed_y

    def optimized_frame_warp(self, frame: np.ndarray,
                             mesh_x: np.ndarray, mesh_y: np.ndarray,
                             motion_x: np.ndarray, motion_y: np.ndarray) -> np.ndarray:
        """
        Optimized frame warping using OpenCV's efficient functions

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

        # Create simplified mapping - sample at fewer points for speed
        sample_factor = 4  # Sample every 4th pixel

        map_x = np.zeros((height // sample_factor, width // sample_factor), dtype=np.float32)
        map_y = np.zeros((height // sample_factor, width // sample_factor), dtype=np.float32)

        mesh_h = height // self.current_mesh_rows
        mesh_w = width // self.current_mesh_cols

        for y in range(0, height, sample_factor):
            for x in range(0, width, sample_factor):
                # Find mesh cell
                cell_i = min(y // mesh_h, self.current_mesh_rows - 1)
                cell_j = min(x // mesh_w, self.current_mesh_cols - 1)

                if cell_i < mesh_x.shape[0] - 1 and cell_j < mesh_x.shape[1] - 1:
                    # Simple nearest neighbor instead of bilinear for speed
                    motion_x_val = motion_x[cell_i, cell_j]
                    motion_y_val = motion_y[cell_i, cell_j]

                    map_x[y // sample_factor, x // sample_factor] = x - motion_x_val
                    map_y[y // sample_factor, x // sample_factor] = y - motion_y_val
                else:
                    map_x[y // sample_factor, x // sample_factor] = x
                    map_y[y // sample_factor, x // sample_factor] = y

        # Upscale mapping to full resolution
        map_x_full = cv2.resize(map_x, (width, height), interpolation=cv2.INTER_LINEAR)
        map_y_full = cv2.resize(map_y, (width, height), interpolation=cv2.INTER_LINEAR)

        # Apply remapping with fast interpolation
        warped_frame = cv2.remap(frame, map_x_full, map_y_full, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

        return warped_frame

    def process_frame_realtime(self, frame: np.ndarray,
                               prev_gray: Optional[np.ndarray] = None,
                               prev_features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process single frame for real-time stabilization

        Args:
            frame: Current frame
            prev_gray: Previous grayscale frame
            prev_features: Previous frame features

        Returns:
            Tuple of (stabilized_frame, current_gray, current_features)
        """
        frame_start_time = time.time()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None or prev_features is None:
            # First frame - just detect features
            features = self.optimized_feature_detection(gray)
            self.processing_stats['avg_features'] = len(features) if features is not None else 0
            return frame, gray, features

        # Feature tracking
        features = self.optimized_feature_detection(gray)
        if len(prev_features) > 0 and len(features) > 0:
            # Track features from previous frame
            tracked_features, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_features, None, **self.lk_params)

            # Filter good tracks
            good_mask = status.flatten() == 1
            if np.sum(good_mask) > 0:
                good_tracked = tracked_features[good_mask]
                good_prev = prev_features[good_mask]

                # Calculate motion magnitude for adaptive mesh
                if len(good_tracked) > 0 and len(good_prev) > 0:
                    if good_tracked.ndim == 3:
                        motion_vectors = good_tracked.squeeze(1) - good_prev.squeeze(1)
                    else:
                        motion_vectors = good_tracked - good_prev

                    motion_magnitude = np.mean(np.sqrt(motion_vectors[:, 0] ** 2 + motion_vectors[:, 1] ** 2))
                    self.motion_history.append(motion_magnitude)

                    # Adapt mesh resolution
                    avg_motion = np.mean(list(self.motion_history))
                    self.adapt_mesh_resolution(avg_motion)

                    # Create adaptive mesh
                    height, width = gray.shape
                    mesh_x, mesh_y = self.create_adaptive_mesh(height, width)

                    # Motion propagation (parallel if enabled)
                    motion_x, motion_y = self.parallel_motion_propagation(
                        good_prev, good_tracked, mesh_x, mesh_y)

                    # Fast temporal smoothing
                    smooth_x, smooth_y = self.fast_temporal_smoothing(motion_x, motion_y)

                    # Optimized warping
                    stabilized_frame = self.optimized_frame_warp(frame, mesh_x, mesh_y, smooth_x, smooth_y)

                    # Update statistics
                    self.processing_stats['avg_features'] = len(good_tracked)
                else:
                    stabilized_frame = frame
            else:
                stabilized_frame = frame
        else:
            stabilized_frame = frame

        # Track performance
        frame_time = time.time() - frame_start_time
        self.frame_times.append(frame_time)

        return stabilized_frame, gray, features

    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        if len(self.frame_times) > 0:
            avg_frame_time = np.mean(list(self.frame_times))
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        else:
            avg_frame_time = 0
            current_fps = 0

        return {
            'current_fps': current_fps,
            'avg_frame_time_ms': avg_frame_time * 1000,
            'current_mesh_size': f"{self.current_mesh_rows}x{self.current_mesh_cols}",
            'mesh_adaptations': self.processing_stats['mesh_adaptations'],
            'parallel_operations': self.processing_stats['parallel_ops'],
            'avg_features_tracked': self.processing_stats['avg_features'],
            'temporal_window_size': len(self.vertex_buffer_x)
        }

    def stabilize_video_realtime(self, input_path: str, output_path: str) -> Dict:
        """
        Real-time video stabilization with optimizations

        Args:
            input_path: Input video path
            output_path: Output video path

        Returns:
            Performance results dictionary
        """
        print(f"Starting REAL-TIME MeshFlow stabilization...")
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
        print(f"Optimizations: Adaptive mesh, Parallel={self.enable_parallel}, Window={self.temporal_window}")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Processing variables
        prev_gray = None
        prev_features = None
        frame_count = 0
        total_start_time = time.time()

        print("Processing frames in real-time mode...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            stabilized_frame, curr_gray, curr_features = self.process_frame_realtime(
                frame, prev_gray, prev_features)

            # Write stabilized frame
            out.write(stabilized_frame)

            # Update for next iteration
            prev_gray = curr_gray
            prev_features = curr_features
            frame_count += 1

            # Show progress every 30 frames
            if frame_count % 30 == 0:
                stats = self.get_performance_stats()
                print(f"Frame {frame_count}/{total_frames} - "
                      f"FPS: {stats['current_fps']:.1f}, "
                      f"Mesh: {stats['current_mesh_size']}, "
                      f"Features: {stats['avg_features_tracked']}")

        # Cleanup
        cap.release()
        out.release()

        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        # Final statistics
        total_time = time.time() - total_start_time
        final_stats = self.get_performance_stats()

        results = {
            'total_frames': frame_count,
            'total_processing_time': total_time,
            'average_fps': frame_count / total_time,
            'optimization_stats': final_stats,
            'speedup_factor': (frame_count / total_time) / 0.6  # Compare to original 0.6 FPS
        }

        print(f"\nReal-time stabilization completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average FPS: {results['average_fps']:.2f}")
        print(f"Speedup factor: {results['speedup_factor']:.1f}x")
        print(f"Mesh adaptations: {final_stats['mesh_adaptations']}")

        return results


def compare_original_vs_optimized(test_video_path: str) -> Dict:
    """
    Compare original MeshFlow vs optimized version

    Args:
        test_video_path: Path to test video

    Returns:
        Comparison results
    """
    print("\n" + "=" * 60)
    print("COMPARING ORIGINAL vs OPTIMIZED MESHFLOW")
    print("=" * 60)

    from meshflow_stabilizer import MeshFlowStabilizer

    # Test original version
    print("\n1. Testing ORIGINAL MeshFlow...")
    original_stabilizer = MeshFlowStabilizer(mesh_rows=16, mesh_cols=16)

    start_time = time.time()
    original_results = original_stabilizer.stabilize_video(
        test_video_path, "comparison_original.mp4")
    original_time = time.time() - start_time

    # Test optimized version
    print("\n2. Testing OPTIMIZED MeshFlow...")
    optimized_stabilizer = RealTimeMeshFlowStabilizer(
        base_mesh_size=8, max_mesh_size=16, enable_parallel=True)

    start_time = time.time()
    optimized_results = optimized_stabilizer.stabilize_video_realtime(
        test_video_path, "comparison_optimized.mp4")
    optimized_time = time.time() - start_time

    # Compare results
    comparison = {
        'original': {
            'fps': original_results.get('average_fps', 0),
            'time': original_time,
            'stability_score': original_results.get('average_stability_score', 0)
        },
        'optimized': {
            'fps': optimized_results.get('average_fps', 0),
            'time': optimized_time,
            'speedup': optimized_results.get('speedup_factor', 0)
        },
        'improvement': {
            'fps_improvement': optimized_results.get('average_fps', 0) / max(0.1,
                                                                             original_results.get('average_fps', 0.1)),
            'time_reduction': original_time / max(0.1, optimized_time)
        }
    }

    print(f"\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"Original FPS: {comparison['original']['fps']:.2f}")
    print(f"Optimized FPS: {comparison['optimized']['fps']:.2f}")
    print(f"FPS Improvement: {comparison['improvement']['fps_improvement']:.1f}x")
    print(f"Time Reduction: {comparison['improvement']['time_reduction']:.1f}x")

    return comparison


def main():
    """Main function for real-time optimization research"""
    print("Real-time MeshFlow Optimization Research")
    print("=" * 50)

    # Create test video if needed
    test_video = "realtime_test_video.mp4"
    if not os.path.exists(test_video):
        print("Creating test video...")
        create_test_video_for_realtime(test_video)

    # Run comparison
    comparison_results = compare_original_vs_optimized(test_video)

    # Save research results
    import json
    with open("realtime_optimization_results.json", 'w') as f:
        json.dump(comparison_results, f, indent=2)

    print("\nResearch completed! Results saved to realtime_optimization_results.json")


def create_test_video_for_realtime(output_path: str):
    """Create a test video for real-time evaluation"""
    import os

    width, height = 640, 480
    fps = 30
    duration_frames = 150  # 5 seconds

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx in range(duration_frames):
        time_factor = frame_idx / fps

        # Create scene with various objects
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40

        # Add grid for tracking
        for i in range(0, width, 50):
            cv2.line(frame, (i, 0), (i, height), (100, 100, 100), 1)
        for i in range(0, height, 50):
            cv2.line(frame, (0, i), (width, i), (100, 100, 100), 1)

        # Moving objects
        circle_x = int(width * 0.3 + 100 * np.sin(time_factor * 2))
        circle_y = int(height * 0.5 + 50 * np.cos(time_factor * 1.5))
        cv2.circle(frame, (circle_x, circle_y), 20, (0, 255, 0), -1)

        # Apply realistic camera shake
        shake_x = int(15 * np.sin(time_factor * 5) + 8 * np.random.randn())
        shake_y = int(12 * np.cos(time_factor * 7) + 6 * np.random.randn())

        M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
        shaky_frame = cv2.warpAffine(frame, M, (width, height))

        out.write(shaky_frame)

    out.release()
    print(f"Test video created: {output_path}")


if __name__ == "__main__":
    import os

    main()