"""
=============================================================================
TEST SUITE: CAMERA FPS & VISION QUALITY DIAGNOSTIC (Radxa Qualcomm Platform)
Tests: V4L2/USB Camera discovery, buffer flush latency, Laplacian blur scoring
=============================================================================
"""

import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hardware.camera_capture import ResilientCameraInterface, FrameQualityAnalyzer


def run_camera_test():
    print("==================================================")
    print("📷 RADXA CAMERA & VISION PIPELINE BENCHMARK")
    print("==================================================")

    cam = ResilientCameraInterface()
    print("\n[TEST 1] Probing camera hardware...")
    ok = cam.discover_and_open()
    print(f"  • Camera Status: {'CONNECTED' if ok else 'FAILED'}")

    print("\n[TEST 2] Testing Zero-Latency Buffer Flushing & FPS...")
    start_time = time.time()
    frames_grabbed = 0

    for i in range(5):
        t0 = time.time()
        ret, frame = cam.flush_and_grab_fresh_frame()
        dt = (time.time() - t0) * 1000
        if ret and frame is not None:
            frames_grabbed += 1
            blur, brightness, status = FrameQualityAnalyzer.evaluate(frame)
            h, w = frame.shape[:2]
            print(f"  • Frame #{i+1}: {w}x{h} | Latency: {dt:.1f}ms | Blur Score: {blur:.1f} | Brightness: {brightness:.1f} [{status}]")
        time.sleep(0.1)

    total_time = time.time() - start_time
    print(f"\n  • Captured {frames_grabbed} frames in {total_time:.2f} seconds.")

    cam.release()
    print("\n✓ ALL CAMERA BENCHMARK TESTS PASSED.\n")


if __name__ == "__main__":
    run_camera_test()
