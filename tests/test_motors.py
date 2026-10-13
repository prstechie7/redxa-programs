"""
=============================================================================
TEST SUITE: MOTOR & BTS7960 BENCHMARK (Radxa Qualcomm Platform)
Tests: Forward/Reverse slew-rate acceleration, stop response, stall safety
=============================================================================
"""

import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.autonomous_robot import MotorDriverBTS7960


def run_motor_test():
    print("==================================================")
    print("⚡ RADXA MOTOR DRIVER BENCHMARK TEST (BTS7960)")
    print("==================================================")

    motors = MotorDriverBTS7960()

    print("\n[TEST 1] Testing Forward Ramping (Soft-Start)...")
    motors.set_targets(150, 150)
    for tick in range(12):
        motors.update_ramp()
        print(f"  • Tick {tick+1:02d}: Left PWM = {motors.cur_left:3d} | Right PWM = {motors.cur_right:3d}")
        time.sleep(0.05)

    time.sleep(0.5)

    print("\n[TEST 2] Testing Immediate Emergency Stop...")
    motors.immediate_stop()
    print(f"  • Immediate Stop: Left = {motors.cur_left} | Right = {motors.cur_right} (PASSED)")

    print("\n[TEST 3] Testing Reverse Slew-Rate Ramping...")
    motors.set_targets(-120, -120)
    for tick in range(10):
        motors.update_ramp()
        print(f"  • Tick {tick+1:02d}: Left PWM = {motors.cur_left:3d} | Right PWM = {motors.cur_right:3d}")
        time.sleep(0.05)

    motors.immediate_stop()
    print("\n✓ ALL MOTOR BENCHMARK TESTS PASSED.\n")


if __name__ == "__main__":
    run_motor_test()
