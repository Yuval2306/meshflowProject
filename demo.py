#!/usr/bin/env python3
"""
Demo script for MeshFlow Video Stabilization
Simple example showing how to use the stabilization algorithm

Authors: Mor Bone (318408465), Yuval Buker (211854062)
Tel-Hai Academic College - Computer Vision Course (199615)
"""

import os
import cv2
import numpy as np
from meshflow_stabilizer import MeshFlowStabilizer


def create_sample_video(output_path: str, duration_seconds: int = 10):
    """
    Create a sample shaky video for testing

    Args:
        output_path: Path to save the sample video
        duration_seconds: Duration of the video in seconds
    """
    print("Creating sample shaky video for testing...")

    # Video parameters
    fps = 30
    width, height = 640, 480
    total_frames = duration_seconds * fps

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create a simple scene with moving objects
    for frame_idx in range(total_frames):
        # Create a blue background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        frame[:, :, 0] = 100  # Blue channel

        # Add some geometric shapes that move predictably
        time_factor = frame_idx / fps

        # Moving circle
        circle_x = int(width * 0.3 + 100 * np.sin(time_factor * 2))
        circle_y = int(height * 0.5 + 50 * np.cos(time_factor * 1.5))
        cv2.circle(frame, (circle_x, circle_y), 30, (0, 255, 0), -1)

        # Moving rectangle
        rect_x = int(width * 0.7 - 80 * np.sin(time_factor * 1.2))
        rect_y = int(height * 0.3 + 40 * np.sin(time_factor * 0.8))
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 60, rect_y + 40), (0, 0, 255), -1)

        # Add some texture (random dots)
        for _ in range(20):
            dot_x = np.random.randint(0, width)
            dot_y = np.random.randint(0, height)
            cv2.circle(frame, (dot_x, dot_y), 2, (255, 255, 255), -1)

        # Add camera shake (this simulates the instability we want to correct)
        shake_x = int(20 * np.sin(time_factor * 8) + 10 * np.random.randn())
        shake_y = int(15 * np.cos(time_factor * 12) + 8 * np.random.randn())

        # Apply shake by translating the frame
        M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
        shaky_frame = cv2.warpAffine(frame, M, (width, height))

        out.write(shaky_frame)

    out.release()
    print(f"Sample video created: {output_path}")


def run_stabilization_demo():
    """
    Run a complete demonstration of the MeshFlow stabilization
    """
    print("=" * 60)
    print("MeshFlow Video Stabilization Demo")
    print("=" * 60)

    # Create directories
    os.makedirs("demo_videos", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # File paths
    sample_video = "demo_videos/sample_shaky_video.mp4"
    stabilized_video = "results/stabilized_video.mp4"

    # Step 1: Create sample video if it doesn't exist
    if not os.path.exists(sample_video):
        create_sample_video(sample_video, duration_seconds=15)

    # Step 2: Initialize the stabilizer
    print("\nInitializing MeshFlow stabilizer...")
    stabilizer = MeshFlowStabilizer(
        mesh_rows=12,  # Using smaller mesh for demo
        mesh_cols=12,
        visualize=False,  # Set to True to see visualization
        temporal_smooth_radius=8
    )

    # Step 3: Run stabilization
    print("\nStarting video stabilization...")
    try:
        results = stabilizer.stabilize_video(sample_video, stabilized_video)

        # Step 4: Display results
        print("\n" + "=" * 50)
        print("STABILIZATION RESULTS")
        print("=" * 50)
        print(f"✓ Input video: {sample_video}")
        print(f"✓ Output video: {stabilized_video}")
        print(f"✓ Total frames processed: {results['total_frames']}")
        print(f"✓ Processing time: {results['processing_time']:.2f} seconds")
        print(f"✓ Average FPS: {results['average_fps']:.2f}")
        print(f"✓ Stability score: {results['average_stability_score']:.3f}")
        print(f"✓ Average motion magnitude: {results['average_motion_magnitude']:.2f}")

        # Step 5: Create a comparison video (side by side)
        create_comparison_video(sample_video, stabilized_video)

        print("\n🎉 Demo completed successfully!")
        print(f"📁 Check the 'results/' directory for output files")

    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        raise


def create_comparison_video(original_path: str, stabilized_path: str):
    """
    Create a side-by-side comparison video of original vs stabilized

    Args:
        original_path: Path to original video
        stabilized_path: Path to stabilized video
    """
    print("\nCreating side-by-side comparison video...")

    # Open both videos
    cap_orig = cv2.VideoCapture(original_path)
    cap_stab = cv2.VideoCapture(stabilized_path)

    if not cap_orig.isOpened() or not cap_stab.isOpened():
        print("Warning: Could not open videos for comparison")
        return

    # Get video properties
    fps = int(cap_orig.get(cv2.CAP_PROP_FPS))
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video writer for comparison
    comparison_path = "results/comparison_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(comparison_path, fourcc, fps, (width * 2, height + 60))

    frame_count = 0
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_stab, frame_stab = cap_stab.read()

        if not ret_orig or not ret_stab:
            break

        # Create comparison frame
        comparison_frame = np.ones((height + 60, width * 2, 3), dtype=np.uint8) * 255

        # Add labels
        cv2.putText(comparison_frame, "Original (Shaky)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(comparison_frame, "MeshFlow Stabilized", (width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Place frames
        comparison_frame[60:60 + height, 0:width] = frame_orig
        comparison_frame[60:60 + height, width:2 * width] = frame_stab

        # Add frame number
        cv2.putText(comparison_frame, f"Frame: {frame_count}", (10, height + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        out.write(comparison_frame)
        frame_count += 1

    cap_orig.release()
    cap_stab.release()
    out.release()

    print(f"✓ Comparison video created: {comparison_path}")


def analyze_stabilization_performance(stabilizer):
    """
    Analyze and display performance metrics

    Args:
        stabilizer: MeshFlowStabilizer instance with performance data
    """
    metrics = stabilizer.performance_metrics

    if not metrics['processing_times']:
        print("No performance data available")
        return

    print("\n" + "=" * 40)
    print("PERFORMANCE ANALYSIS")
    print("=" * 40)

    # Processing time analysis
    avg_time = np.mean(metrics['processing_times'])
    min_time = np.min(metrics['processing_times'])
    max_time = np.max(metrics['processing_times'])

    print(f"Processing time per frame:")
    print(f"  Average: {avg_time:.4f} seconds")
    print(f"  Min: {min_time:.4f} seconds")
    print(f"  Max: {max_time:.4f} seconds")

    # Motion analysis
    if metrics['motion_vectors']:
        avg_motion = np.mean(metrics['motion_vectors'])
        print(f"Average motion magnitude: {avg_motion:.2f} pixels")

    # Stability analysis
    if metrics['stabilization_quality']:
        avg_stability = np.mean(metrics['stabilization_quality'])
        print(f"Average stability score: {avg_stability:.3f}")


def test_different_configurations():
    """
    Test different MeshFlow configurations to compare performance
    """
    print("\n" + "=" * 50)
    print("TESTING DIFFERENT CONFIGURATIONS")
    print("=" * 50)

    sample_video = "demo_videos/sample_shaky_video.mp4"

    if not os.path.exists(sample_video):
        print("Sample video not found. Creating one...")
        create_sample_video(sample_video, duration_seconds=10)

    configurations = [
        {"name": "Small Mesh", "mesh_rows": 8, "mesh_cols": 8, "temporal_smooth_radius": 5},
        {"name": "Medium Mesh", "mesh_rows": 16, "mesh_cols": 16, "temporal_smooth_radius": 10},
        {"name": "Large Mesh", "mesh_rows": 24, "mesh_cols": 24, "temporal_smooth_radius": 15},
    ]

    results = []

    for config in configurations:
        print(f"\nTesting configuration: {config['name']}")

        # Initialize stabilizer with current config
        stabilizer = MeshFlowStabilizer(
            mesh_rows=config['mesh_rows'],
            mesh_cols=config['mesh_cols'],
            temporal_smooth_radius=config['temporal_smooth_radius']
        )

        output_path = f"results/stabilized_{config['name'].lower().replace(' ', '_')}.mp4"

        try:
            result = stabilizer.stabilize_video(sample_video, output_path)
            result['config_name'] = config['name']
            results.append(result)

            print(f"✓ {config['name']}: {result['average_fps']:.1f} FPS, "
                  f"Stability: {result['average_stability_score']:.3f}")

        except Exception as e:
            print(f"❌ {config['name']} failed: {str(e)}")

    # Compare results
    if results:
        print("\n" + "=" * 60)
        print("CONFIGURATION COMPARISON")
        print("=" * 60)
        print(f"{'Configuration':<15} {'FPS':<8} {'Stability':<10} {'Time':<8}")
        print("-" * 60)

        for result in results:
            print(f"{result['config_name']:<15} "
                  f"{result['average_fps']:<8.1f} "
                  f"{result['average_stability_score']:<10.3f} "
                  f"{result['processing_time']:<8.1f}")


def main():
    """
    Main demo function
    """
    try:
        # Run basic demo
        run_stabilization_demo()

        # Test different configurations
        test_different_configurations()

        print("\n" + "🎯" + "=" * 58 + "🎯")
        print("📋 DEMO SUMMARY")
        print("🎯" + "=" * 58 + "🎯")
        print("✅ MeshFlow implementation completed")
        print("✅ Sample videos created and processed")
        print("✅ Performance metrics calculated")
        print("✅ Multiple configurations tested")
        print("✅ Comparison videos generated")
        print("\n📁 All results saved in 'results/' directory")
        print("📺 View the comparison video to see stabilization effect")

    except KeyboardInterrupt:
        print("\n⏸️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()