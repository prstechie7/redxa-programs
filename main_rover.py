"""
=============================================================================
MASTER AGRIBOT BOOT SCRIPT FOR RADXA QUALCOMM EDGE HARDWARE
Platform: Radxa Qualcomm Edge SBC (Debian / Linux aarch64)
Role: Central System Orchestrator (Spawns Drive, Vision, Telemetry & SOS Daemons)
=============================================================================
"""

import os
import sys
import json
import time
import threading
import logging

from core.autonomous_robot import run_robot_loop
from hardware.camera_capture import start_camera_stream
from hardware.sensor_telemetry import start_telemetry_loop
from hardware.emergency_calling_agent import AgriBotCallingAgent

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [RADXA_CORE] %(message)s"
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_hardware_config():
    """Loads central configuration settings."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                logging.info(f"Loaded hardware configuration for: {cfg.get('platform', 'Radxa SBC')}")
                return cfg
        except Exception as e:
            logging.warning(f"Failed to parse config.json: {e}")
    return {}


def main():
    logging.info("==================================================")
    logging.info("⚡ BOOTING AGRIBOT ON RADXA QUALCOMM EDGE SBC ⚡")
    logging.info("==================================================")

    config = load_hardware_config()

    # Initialize Emergency Cellular Calling Agent
    emergency_cfg = config.get("emergency_sos", {})
    calling_agent = AgriBotCallingAgent(
        port=emergency_cfg.get("modem_uart_port", "/dev/ttyUSB0"),
        baudrate=emergency_cfg.get("modem_baudrate", 9600),
        default_phone=emergency_cfg.get("emergency_phone", "+918319556016")
    )

    # 1. Start Sensor & Soil Telemetry Thread
    logging.info("Spawning Soil Moisture & Climate Telemetry Thread...")
    t_sensors = threading.Thread(target=start_telemetry_loop, daemon=True)
    t_sensors.start()

    # 2. Start Low-Latency Computer Vision Stream
    logging.info("Spawning Low-Latency USB/V4L2 Camera Inspection Thread...")
    t_camera = threading.Thread(target=start_camera_stream, daemon=True)
    t_camera.start()

    # 3. Start Drive & Obstacle Watchdog (Runs in Foreground)
    logging.info("Engaging Chassis Locomotion & Obstacle Watchdog Engine...")
    try:
        run_robot_loop(calling_agent=calling_agent)
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Halting rover safely...")
    except Exception as e:
        logging.error(f"Rover runtime encountered error: {e}")
        calling_agent.trigger_incident("ROVER_STUCK")
    finally:
        logging.info("AgriBot Core shutdown sequence concluded.")


if __name__ == "__main__":
    main()
