# 🌾 AgriBot Autonomous Platform for Radxa Qualcomm SBC (v2.0 PRO)

An end-to-end, industrial-grade precision agriculture and autonomous rover platform optimized for **Radxa Edge Hardware powered by Qualcomm SoCs** (e.g. Snapdragon, QCS610, QRB5165, Radxa Edge SBC running Debian Linux / Radxa OS aarch64).

---

## 📂 Repository Architecture (`redxa-programs` layout)

```text
redxa-programs/ (https://github.com/prstechie7/ml-model)
├── README.md                      # Hardware pinouts, setup, and quickstart
├── requirements.txt               # Onboard Python dependencies (aarch64 Linux)
├── config.json                    # Central pinout, thresholds & baudrate config
├── main_rover.py                  # Master system orchestrator (boots all services)
│
├── core/                          # Drive & Navigation
│   ├── __init__.py
│   ├── autonomous_robot.py        # BTS7960 motor control, RC arcade steering & braking
│   ├── gps_navigator.py           # NEO-M8N UART GPS parsing & field waypoint tracking
│   └── drone_mesh_bridge.py       # Aerial-to-ground MQTT scouting & dynamic route injector
│
├── ai/                            # Edge Vision & Diagnostic Models
│   ├── __init__.py
│   ├── edge_ai_inference.py       # ONNX runtime engine (runs YOLO11n + MobileNetV2)
│   └── models/                    # Quantized model weights (.onnx)
│       ├── yolo11n_pest.onnx      # Quantized edge pest detector
│       └── mobilenetv2_leaf.onnx  # Offline crop disease classifier
│
├── hardware/                      # Sensor & Actuator Drivers
│   ├── __init__.py
│   ├── camera_capture.py          # Low-latency camera capture & Laplacian blur filter
│   ├── sensor_telemetry.py        # 3-depth soil probe ADC & DHT11 temperature/humidity
│   ├── emergency_calling_agent.py # SIM800L UART voice dialing & GPS SMS dispatch
│   ├── precision_sprayer.py       # 12V diaphragm pump/solenoid spot-sprayer actuator
│   └── power_solar_monitor.py     # INA219 I2C 3S Li-ion battery health & solar harvest
│
├── tests/                         # Hardware Benchmarking & Pin Tests
│   ├── __init__.py
│   ├── test_motors.py             # Forward/reverse ramp & stall current verification
│   ├── test_sensors.py            # Raw ADC reading & timing verification
│   └── test_camera.py             # FPS, latency & auto-exposure diagnostic check
│
└── systemd/                       # Auto-Boot on Battery Power
    └── agribot.service            # Linux service to run main_rover.py on boot
```

---

## 📄 Component Responsibilities on the Radxa Board

| Component | File | Responsibilities on the Radxa Board |
| :--- | :--- | :--- |
| **Orchestrator** | `main_rover.py` | Spawns drive, camera, sensor, and emergency calling threads; monitors watchdog health. |
| **Drive System** | `core/autonomous_robot.py` | Brownout-prevention acceleration ramping, RC FS-i6X decoding, and ultrasonic emergency braking. |
| **GIS Navigation** | `core/gps_navigator.py` | NEO-M8N NMEA sentence decoding, Haversine lawnmower swath grid tracking, and Geofence boundary brake. |
| **AI Inference** | `ai/edge_ai_inference.py` | Takes frames directly from `camera_capture.py` and runs offline ONNX models in <20ms. |
| **Vision Feed** | `hardware/camera_capture.py` | Threaded V4L2/USB capture without OpenCV queue lag; rejects blurry/overexposed leaves. |
| **Soil & Climate** | `hardware/sensor_telemetry.py` | Reads calibrated ADC soil moisture at 3 depths, DHT11 temp/humidity, and logs to cloud/CSV. |
| **Emergency SOS** | `hardware/emergency_calling_agent.py` | Dials farmer phone via SIM800L and sends SMS coordinates when rover gets stuck or triggers low battery. |
| **Spot Sprayer** | `hardware/precision_sprayer.py` | Drives 12V pump & solenoid valves only on detected pest/disease coordinates (eliminates 40%+ chemical waste). |
| **Solar & Power** | `hardware/power_solar_monitor.py` | INA219 I2C monitor for 3S Li-ion battery pack and solar charging, triggering Return-to-Base at <15% SoC. |

---

## ⚙️ Pinout & Hardware Configuration (`config.json`)

The platform is centrally configured through `config.json` without modifying code:

| Subsystem | Parameter | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **Left BTS7960** | `left_rpwm`, `left_lpwm` | `GPIO 25`, `GPIO 26` | Left motor forward & backward PWM |
| **Right BTS7960** | `right_rpwm`, `right_lpwm` | `GPIO 27`, `GPIO 14` | Right motor forward & backward PWM |
| **Ultrasonic** | `ultrasonic_trig`, `echo` | `GPIO 32`, `GPIO 33` | HC-SR04 obstacle avoidance |
| **GNSS GPS** | `gps_uart_port`, `baudrate`| `/dev/ttyAMA0`, `9600` | u-blox NEO-M8N UART port |
| **Camera** | `camera_device`, `fallback`| `/dev/v4l/by-id/...`, `0` | Stable V4L2 USB camera node |
| **Soil Probes** | `adc_channels` | `[34, 35, 32]` | 3-depth capacitive moisture sensors |
| **Actuators** | `pump_relay`, `solenoid` | `GPIO 12`, `GPIO 13` | 12V pump relay & spot valve |
| **GSM Modem** | `modem_uart_port` | `/dev/ttyUSB0` | SIM800L / A9G cellular modem |

---

## 🚀 Auto-Start on Battery Power (`systemd/agribot.service`)

To enable the rover to start automatically as soon as battery power is switched on in the field:

```bash
# 1. Copy service file to systemd directory
sudo cp systemd/agribot.service /etc/systemd/system/

# 2. Reload systemd daemons
sudo systemctl daemon-reload

# 3. Enable service to run on boot
sudo systemctl enable agribot.service

# 4. Start service manually or check status
sudo systemctl start agribot.service
sudo systemctl status agribot.service
```

---

## 🧪 Hardware Benchmarking & Validation (`tests/`)

Run tests individually on the Radxa SBC before deploying into the field:

```bash
# Test 1: Motor forward/reverse slew-rate ramp & stall protection
python3 tests/test_motors.py

# Test 2: Soil moisture ADC channels and DHT11 reading
python3 tests/test_sensors.py

# Test 3: Camera frame capture latency, buffer flush & blur analysis
python3 tests/test_camera.py
```

---

## ⚡ Master Orchestrator Execution

To run the complete system manually:

```bash
# Install onboard dependencies
pip install -r requirements.txt

# Start master rover orchestrator
python3 main_rover.py
```
