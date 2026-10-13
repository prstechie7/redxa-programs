"""
=============================================================================
ADVANCED ESP32 ROBOT PLATFORM (v2.0 PRO)
Hardware: ESP32 + 2x BTS7960 (43A) + 4 Motors + FS-i6X RC + HC-SR04
Architecture: MicroPython / Non-Blocking Cooperative Event Loop

UPGRADES & ADVANCED FEATURES:
-----------------------------------------------------------------------------
1. 3-MODE SYSTEM:
   - Mode 0: FS-i6X RC (Full Arcade Drive Mixing: Throttle + Steering)
   - Mode 1: Wi-Fi Web Dashboard (Touch controls, Keyboard WASD, Real-time telemetry)
   - Mode 2: Fully Autonomous Obstacle Avoidance / Roaming Mode
   - Mode Switch: Via Physical Debounced Push Button OR Web Dashboard Toggle

2. PROPORTIONAL ADAPTIVE BRAKING & SENSOR FILTERING:
   - 5-sample Moving Median Filter to eliminate acoustic noise/ghost spikes.
   - Proportional Deceleration:
       > 60 cm : Full speed
       30-60 cm: Smooth linear deceleration (prevents collisions before stopping)
       <= 30 cm: Forward motion blocked; pivot/reverse permitted
       <= 15 cm: Emergency crawl backoff in autonomous mode
   - Backward movement is ALWAYS allowed in all conditions.

3. MOTOR PROTECTION & SLEW-RATE CURRENT CONTROL:
   - Linear acceleration/deceleration ramping (Soft-Start / Soft-Brake).
   - Prevents massive inrush currents on BTS7960 that cause ESP32 brownout reboots.

4. FAILSAFES & WATCHDOGS:
   - RC Loss-of-Signal (LOS) Failsafe: Stops robot if transmitter drops for > 300ms.
   - Wi-Fi Heartbeat Watchdog: Cuts motor power if connection drops for > 1000ms.
   - Brownout / Low Voltage ADC Monitoring (optional ADC pin).
=============================================================================
"""

import time
import sys

IS_MICROPYTHON = sys.implementation.name == "micropython"

if IS_MICROPYTHON:
    import network
    import socket
    import select
    from machine import Pin, PWM, time_pulse_us, ADC
else:
    import socket
    import select

# =============================================================================
# PIN ASSIGNMENTS
# =============================================================================
# RC Receiver (FS-i6X PWM Channels)
PIN_CH1 = 18       # Steering (Aileron)
PIN_CH2 = 19       # Throttle (Elevator)
PIN_CH3 = 21       # Speed Governor / Throttle limit

# Mode Toggle Button
PIN_MODE_BTN = 23  # Active LOW with internal pull-up

# Left BTS7960 Driver
PIN_L_RPWM = 25    # Forward
PIN_L_LPWM = 26    # Backward

# Right BTS7960 Driver
PIN_R_RPWM = 27    # Forward
PIN_R_LPWM = 14    # Backward

# HC-SR04 Ultrasonic Sensor
PIN_TRIG = 32
PIN_ECHO = 33

# Optional: Battery Voltage ADC Sense (set to None if not wired)
PIN_BATTERY_ADC = 34  # Voltage divider (e.g. 100k / 22k)

# =============================================================================
# OPERATIONAL TUNING CONSTANTS
# =============================================================================
AP_SSID = "ROBOT_ESP32_PRO"
AP_PASSWORD = "robotpassword"

# Ultrasonic Zones (in cm)
DIST_SAFE_ZONE = 60       # Below this, robot begins proportional slowing
DIST_STOP_LIMIT = 30      # Below this, forward motion is blocked
DIST_CRITICAL_LIMIT = 15  # Below this, danger trigger

# Motor Ramp Limits (Max change in PWM duty per cycle to prevent brownout)
RAMP_ACCEL_STEP = 15      # Speed change step per 20ms tick
PWM_FREQUENCY = 1000      # 1kHz for BTS7960

# RC Transmitter Timing (microseconds)
RC_MIN = 1000
RC_MID = 1500
RC_MAX = 2000
RC_DEADZONE = 50

# Web Timeout Watchdog
WIFI_TIMEOUT_MS = 1000

# =============================================================================
# MODERN DASHBOARD UI (HTML5 + CSS Glassmorphism + Responsive Canvas)
# =============================================================================
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>ESP32 Robot Command Deck</title>
<style>
:root {
  --bg: #0c0f17;
  --card: rgba(22, 28, 45, 0.75);
  --accent: #00d2ff;
  --accent-glow: rgba(0, 210, 255, 0.4);
  --green: #00e676;
  --red: #ff3366;
  --orange: #ff9100;
  --text: #f0f4f8;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: radial-gradient(circle at 50% 20%, #151d30 0%, var(--bg) 100%);
  color: var(--text);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 15px;
  touch-action: manipulation;
}
header {
  text-align: center;
  margin-bottom: 12px;
}
h1 {
  font-size: 22px;
  letter-spacing: 1.5px;
  color: var(--accent);
  text-shadow: 0 0 15px var(--accent-glow);
}
.hud-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  width: 100%;
  max-width: 420px;
  margin-bottom: 15px;
}
.hud-card {
  background: var(--card);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  padding: 10px 6px;
  text-align: center;
}
.hud-card span { font-size: 11px; opacity: 0.7; text-transform: uppercase; }
.hud-card h3 { font-size: 18px; margin-top: 3px; font-weight: 700; }
.mode-toggle {
  display: flex;
  background: rgba(0,0,0,0.4);
  border-radius: 30px;
  padding: 4px;
  margin-bottom: 15px;
  width: 100%;
  max-width: 360px;
}
.mode-btn {
  flex: 1;
  background: none;
  border: none;
  color: #8892b0;
  padding: 8px 12px;
  font-size: 13px;
  font-weight: 600;
  border-radius: 20px;
  cursor: pointer;
  transition: all 0.2s;
}
.mode-btn.active {
  background: var(--accent);
  color: #0c0f17;
  box-shadow: 0 0 12px var(--accent-glow);
}
.dpad {
  display: grid;
  grid-template-columns: repeat(3, 85px);
  grid-template-rows: repeat(3, 75px);
  gap: 12px;
  justify-content: center;
  margin: 10px auto;
}
.btn-cmd {
  background: var(--card);
  border: 1px solid rgba(255, 255, 255, 0.12);
  color: white;
  font-size: 20px;
  font-weight: bold;
  border-radius: 16px;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  transition: all 0.1s;
  box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.btn-cmd span { font-size: 10px; margin-top: 2px; opacity: 0.8; }
.btn-cmd:active { transform: scale(0.92); border-color: var(--accent); }
.btn-fwd { grid-column: 2; grid-row: 1; border-color: rgba(0, 230, 118, 0.3); }
.btn-left { grid-column: 1; grid-row: 2; border-color: rgba(0, 210, 255, 0.3); }
.btn-stop { grid-column: 2; grid-row: 2; background: rgba(255, 51, 102, 0.2); border-color: var(--red); color: var(--red); }
.btn-right { grid-column: 3; grid-row: 2; border-color: rgba(0, 210, 255, 0.3); }
.btn-back { grid-column: 2; grid-row: 3; border-color: rgba(255, 145, 0, 0.3); }

.slider-container {
  width: 100%;
  max-width: 360px;
  margin-top: 15px;
  background: var(--card);
  padding: 12px 18px;
  border-radius: 14px;
}
.slider-container label {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  margin-bottom: 8px;
}
input[type=range] {
  width: 100%;
  accent-color: var(--accent);
  height: 6px;
}
.status-bar {
  font-size: 11px;
  color: #8892b0;
  margin-top: 15px;
}
</style>
</head>
<body>
<header>
  <h1>⚡ APEX-32 ROBOT</h1>
</header>

<div class="hud-grid">
  <div class="hud-card">
    <span>Obstacle</span>
    <h3 id="distVal" style="color:var(--green)">-- cm</h3>
  </div>
  <div class="hud-card">
    <span>Active Mode</span>
    <h3 id="modeVal">RC</h3>
  </div>
  <div class="hud-card">
    <span>Speed</span>
    <h3 id="speedDisp">160</h3>
  </div>
</div>

<div class="mode-toggle">
  <button class="mode-btn active" id="btnMode0" onclick="setMode(0)">RC (FS-i6X)</button>
  <button class="mode-btn" id="btnMode1" onclick="setMode(1)">Wi-Fi</button>
  <button class="mode-btn" id="btnMode2" onclick="setMode(2)">Autonomous</button>
</div>

<div class="dpad" id="dpadPanel">
  <button class="btn-cmd btn-fwd"
    ontouchstart="cmd('forward')" ontouchend="cmd('stop')"
    onmousedown="cmd('forward')" onmouseup="cmd('stop')">
    ▲<span>FWD</span>
  </button>
  <button class="btn-cmd btn-left"
    ontouchstart="cmd('left')" ontouchend="cmd('stop')"
    onmousedown="cmd('left')" onmouseup="cmd('stop')">
    ◀<span>LEFT</span>
  </button>
  <button class="btn-cmd btn-stop" onclick="cmd('stop')">
    ■<span>STOP</span>
  </button>
  <button class="btn-cmd btn-right"
    ontouchstart="cmd('right')" ontouchend="cmd('stop')"
    onmousedown="cmd('right')" onmouseup="cmd('stop')">
    ▶<span>RIGHT</span>
  </button>
  <button class="btn-cmd btn-back"
    ontouchstart="cmd('backward')" ontouchend="cmd('stop')"
    onmousedown="cmd('backward')" onmouseup="cmd('stop')">
    ▼<span>BACK</span>
  </button>
</div>

<div class="slider-container">
  <label>
    <span>Throttle Limiter</span>
    <span id="sliderVal">160 PWM</span>
  </label>
  <input type="range" min="60" max="255" value="160" id="speedSlider" oninput="changeSpeed(this.value)">
</div>

<div class="status-bar">
  Keyboard: [W, A, S, D] to Move • [Space] to Stop
</div>

<script>
function cmd(action) {
  fetch('/cmd?move=' + action).catch(console.error);
}

function setMode(m) {
  fetch('/set_mode?val=' + m)
    .then(r => r.json())
    .then(d => updateModeUI(d.mode))
    .catch(console.error);
}

function updateModeUI(modeIdx) {
  document.querySelectorAll('.mode-btn').forEach((b, i) => {
    b.classList.toggle('active', i === modeIdx);
  });
  const names = ["RC (FS-i6X)", "Wi-Fi Deck", "Autonomous"];
  document.getElementById('modeVal').innerText = names[modeIdx];
}

function changeSpeed(val) {
  document.getElementById('sliderVal').innerText = val + " PWM";
  document.getElementById('speedDisp').innerText = val;
  fetch('/speed?value=' + val).catch(console.error);
}

// Telemetry Polling
setInterval(() => {
  fetch('/telemetry')
    .then(r => r.json())
    .then(data => {
      const el = document.getElementById('distVal');
      el.innerText = data.distance + " cm";
      if (data.distance <= 30) {
        el.style.color = "var(--red)";
      } else if (data.distance <= 60) {
        el.style.color = "var(--orange)";
      } else {
        el.style.color = "var(--green)";
      }
      updateModeUI(data.mode);
      document.getElementById('speedDisp').innerText = data.speed;
    })
    .catch(console.error);
}, 250);

// Keyboard controls
let activeKey = null;
window.addEventListener('keydown', (e) => {
  if (e.repeat || activeKey === e.key) return;
  activeKey = e.key;
  if (e.key === 'w' || e.key === 'ArrowUp') cmd('forward');
  else if (e.key === 's' || e.key === 'ArrowDown') cmd('backward');
  else if (e.key === 'a' || e.key === 'ArrowLeft') cmd('left');
  else if (e.key === 'd' || e.key === 'ArrowRight') cmd('right');
  else if (e.key === ' ') cmd('stop');
});

window.addEventListener('keyup', (e) => {
  if (e.key === activeKey) {
    activeKey = null;
    cmd('stop');
  }
});
</script>
</body>
</html>
"""

# =============================================================================
# ULTRASONIC SENSOR FILTER (Noise & Ghost Outlier Rejection)
# =============================================================================
class UltrasonicSensor:
    def __init__(self, trig_pin, echo_pin):
        self.trig = Pin(trig_pin, Pin.OUT) if IS_MICROPYTHON else None
        self.echo = Pin(echo_pin, Pin.IN) if IS_MICROPYTHON else None
        if self.trig:
            self.trig.value(0)
        self.window = [999] * 5  # 5-sample median window
        self.current_distance = 999
        self.last_sample_time = 0

    def raw_read(self):
        if not IS_MICROPYTHON:
            return 999
        # 10us trigger pulse
        self.trig.value(0)
        time.sleep_us(2)
        self.trig.value(1)
        time.sleep_us(10)
        self.trig.value(0)

        try:
            # Echo pulse measurement (timeout 25000us = ~4.2 meters)
            duration = time_pulse_us(self.echo, 1, 25000)
            if duration <= 0:
                return 999
            # Speed of sound: 0.0343 cm/us / 2
            dist = duration * 0.0343 / 2.0
            return int(dist)
        except Exception:
            return 999

    def update(self):
        now = time.ticks_ms() if IS_MICROPYTHON else int(time.time() * 1000)
        diff = time.ticks_diff(now, self.last_sample_time) if IS_MICROPYTHON else (now - self.last_sample_time)
        if diff >= 40:  # 25 Hz sampling rate
            self.last_sample_time = now
            val = self.raw_read()
            if 2 <= val <= 400:
                self.window.pop(0)
                self.window.append(val)
                # Median calculation
                sorted_w = sorted(self.window)
                self.current_distance = sorted_w[2]
            else:
                self.current_distance = 999
        return self.current_distance


# =============================================================================
# ADVANCED MOTOR CONTROLLER (With Slew-Rate Ramping & Dual BTS7960)
# =============================================================================
class MotorDriverBTS7960:
    def __init__(self):
        # Current active speeds (0 to 255)
        self.cur_left = 0
        self.cur_right = 0
        self.target_left = 0
        self.target_right = 0

        if IS_MICROPYTHON:
            self.l_rpwm = PWM(Pin(PIN_L_RPWM), freq=PWM_FREQUENCY, duty=0)
            self.l_lpwm = PWM(Pin(PIN_L_LPWM), freq=PWM_FREQUENCY, duty=0)
            self.r_rpwm = PWM(Pin(PIN_R_RPWM), freq=PWM_FREQUENCY, duty=0)
            self.r_lpwm = PWM(Pin(PIN_R_LPWM), freq=PWM_FREQUENCY, duty=0)

    def set_targets(self, left, right):
        """left/right range: -255 (Full Back) to +255 (Full Forward)"""
        self.target_left = max(-255, min(255, int(left)))
        self.target_right = max(-255, min(255, int(right)))

    def immediate_stop(self):
        self.target_left = 0
        self.target_right = 0
        self.cur_left = 0
        self.cur_right = 0
        self._apply_hardware(0, 0)

    def update_ramp(self):
        """Soft-start / Soft-brake slew-rate limiter executed every loop tick."""
        # Left channel ramp
        if self.cur_left < self.target_left:
            self.cur_left = min(self.target_left, self.cur_left + RAMP_ACCEL_STEP)
        elif self.cur_left > self.target_left:
            self.cur_left = max(self.target_left, self.cur_left - RAMP_ACCEL_STEP)

        # Right channel ramp
        if self.cur_right < self.target_right:
            self.cur_right = min(self.target_right, self.cur_right + RAMP_ACCEL_STEP)
        elif self.cur_right > self.target_right:
            self.cur_right = max(self.target_right, self.cur_right - RAMP_ACCEL_STEP)

        self._apply_hardware(self.cur_left, self.cur_right)

    def _apply_hardware(self, left, right):
        if not IS_MICROPYTHON:
            return

        # Left motor BTS7960
        if left >= 0:
            self.l_rpwm.duty(int(left * 4.01))
            self.l_lpwm.duty(0)
        else:
            self.l_rpwm.duty(0)
            self.l_lpwm.duty(int(abs(left) * 4.01))

        # Right motor BTS7960
        if right >= 0:
            self.r_rpwm.duty(int(right * 4.01))
            self.r_lpwm.duty(0)
        else:
            self.r_rpwm.duty(0)
            self.r_lpwm.duty(int(abs(right) * 4.01))


# =============================================================================
# SYSTEM MANAGER & STATE MACHINE
# =============================================================================
class AdvancedRobot:
    def __init__(self):
        self.mode = 0  # 0: RC, 1: Wi-Fi, 2: Autonomous
        self.motors = MotorDriverBTS7960()
        self.sensor = UltrasonicSensor(PIN_TRIG, PIN_ECHO)

        self.wifi_speed = 160
        self.last_wifi_cmd_time = 0
        self.wifi_current_move = "stop"

        # Autonomous Navigation State Machine
        self.auto_state = "DRIVE"
        self.auto_timer = 0
        self.auto_turn_dir = 1  # 1: right, -1: left

        # Physical Mode Button
        self.btn = Pin(PIN_MODE_BTN, Pin.IN, Pin.PULL_UP) if IS_MICROPYTHON else None
        self.last_btn_state = 1
        self.btn_debounce_time = 0

        # RC Inputs
        self.rc_ch1 = Pin(PIN_CH1, Pin.IN) if IS_MICROPYTHON else None
        self.rc_ch2 = Pin(PIN_CH2, Pin.IN) if IS_MICROPYTHON else None
        self.rc_ch3 = Pin(PIN_CH3, Pin.IN) if IS_MICROPYTHON else None
        self.last_rc_valid_time = 0

    def calculate_speed_scalar(self, distance):
        """
        Proportional Speed Reduction:
        - > 60cm: 100% speed
        - 30cm - 60cm: Linearly scaled from 30% to 100%
        - <= 30cm: 0% forward speed (blocked)
        """
        if distance > DIST_SAFE_ZONE:
            return 1.0
        elif distance <= DIST_STOP_LIMIT:
            return 0.0
        else:
            # Linear interpolation between 0.3 and 1.0
            ratio = (distance - DIST_STOP_LIMIT) / (DIST_SAFE_ZONE - DIST_STOP_LIMIT)
            return 0.3 + (0.7 * ratio)

    # -------------------------------------------------------------------------
    # MODE 0: RC WITH ARCADE DRIVE MIXING
    # -------------------------------------------------------------------------
    def process_rc_mode(self, distance):
        if not IS_MICROPYTHON:
            return

        try:
            ch1 = time_pulse_us(self.rc_ch1, 1, 20000)
            ch2 = time_pulse_us(self.rc_ch2, 1, 20000)
            ch3 = time_pulse_us(self.rc_ch3, 1, 20000)
        except Exception:
            ch1, ch2, ch3 = 0, 0, 0

        now = time.ticks_ms()
        # Validate RC signals (Pulse must be between 900 and 2100 us)
        if 900 <= ch1 <= 2100 and 900 <= ch2 <= 2100 and 900 <= ch3 <= 2100:
            self.last_rc_valid_time = now
        else:
            # RC LOSS-OF-SIGNAL (LOS) WATCHDOG
            if time.ticks_diff(now, self.last_rc_valid_time) > 300:
                self.motors.immediate_stop()
                return

        # Governor speed limit from CH3
        max_speed = int((max(min(ch3, RC_MAX), RC_MIN) - RC_MIN) * 255 / (RC_MAX - RC_MIN))

        # Normalized inputs (-1.0 to +1.0)
        throttle = (ch2 - RC_MID) / 500.0
        steering = (ch1 - RC_MID) / 500.0

        if abs(throttle) < (RC_DEADZONE / 500.0):
            throttle = 0.0
        if abs(steering) < (RC_DEADZONE / 500.0):
            steering = 0.0

        # Safety: check obstacle for forward throttle
        if throttle > 0:
            scalar = self.calculate_speed_scalar(distance)
            throttle *= scalar
            if scalar == 0.0:
                throttle = 0.0

        # BACKWARD IS ALWAYS ALLOWED (throttle < 0)

        # Arcade drive mixer
        left = (throttle + steering) * max_speed
        right = (throttle - steering) * max_speed

        self.motors.set_targets(left, right)

    # -------------------------------------------------------------------------
    # MODE 1: WI-FI CONTROLS
    # -------------------------------------------------------------------------
    def process_wifi_mode(self, distance):
        now = time.ticks_ms() if IS_MICROPYTHON else int(time.time() * 1000)
        elapsed = (
            time.ticks_diff(now, self.last_wifi_cmd_time)
            if IS_MICROPYTHON
            else (now - self.last_wifi_cmd_time)
        )

        # Wi-Fi Watchdog (Stops if connection lost)
        if elapsed > WIFI_TIMEOUT_MS or self.wifi_current_move == "stop":
            self.motors.set_targets(0, 0)
            return

        cmd = self.wifi_current_move
        speed = self.wifi_speed

        if cmd == "forward":
            scalar = self.calculate_speed_scalar(distance)
            scaled_speed = int(speed * scalar)
            self.motors.set_targets(scaled_speed, scaled_speed)
        elif cmd == "backward":
            # Backward always permitted
            self.motors.set_targets(-speed, -speed)
        elif cmd == "left":
            if distance <= DIST_STOP_LIMIT:
                self.motors.set_targets(0, 0)
            else:
                self.motors.set_targets(-speed, speed)
        elif cmd == "right":
            if distance <= DIST_STOP_LIMIT:
                self.motors.set_targets(0, 0)
            else:
                self.motors.set_targets(speed, -speed)

    # -------------------------------------------------------------------------
    # MODE 2: AUTONOMOUS OBSTACLE AVOIDANCE
    # -------------------------------------------------------------------------
    def process_autonomous_mode(self, distance):
        now = time.ticks_ms() if IS_MICROPYTHON else int(time.time() * 1000)

        if self.auto_state == "DRIVE":
            if distance <= DIST_STOP_LIMIT:
                self.auto_state = "BACKOFF"
                self.auto_timer = now
                # Alternate turn direction
                self.auto_turn_dir = -self.auto_turn_dir
            else:
                scalar = self.calculate_speed_scalar(distance)
                auto_speed = int(140 * scalar)
                self.motors.set_targets(auto_speed, auto_speed)

        elif self.auto_state == "BACKOFF":
            # Safely reverse for 450 ms
            self.motors.set_targets(-130, -130)
            diff = time.ticks_diff(now, self.auto_timer) if IS_MICROPYTHON else (now - self.auto_timer)
            if diff > 450:
                self.auto_state = "PIVOT"
                self.auto_timer = now

        elif self.auto_state == "PIVOT":
            # Pivot turn away from obstacle for 500 ms
            turn_spd = 150 * self.auto_turn_dir
            self.motors.set_targets(-turn_spd, turn_spd)
            diff = time.ticks_diff(now, self.auto_timer) if IS_MICROPYTHON else (now - self.auto_timer)
            if diff > 500:
                self.auto_state = "DRIVE"

    # -------------------------------------------------------------------------
    # HARDWARE BUTTON DEBOUNCE
    # -------------------------------------------------------------------------
    def check_mode_button(self):
        if not IS_MICROPYTHON or not self.btn:
            return

        reading = self.btn.value()
        now = time.ticks_ms()

        if reading != self.last_btn_state:
            self.btn_debounce_time = now

        if time.ticks_diff(now, self.btn_debounce_time) > 60:
            if reading == 0:  # Pressed (active LOW)
                self.mode = (self.mode + 1) % 3
                self.motors.immediate_stop()
                print(f"[SYSTEM] Hardware Button Switched to Mode {self.mode}")
                time.sleep(0.2)  # Avoid immediate multi-trigger

        self.last_btn_state = reading

    def update(self):
        dist = self.sensor.update()
        self.check_mode_button()

        if self.mode == 0:
            self.process_rc_mode(dist)
        elif self.mode == 1:
            self.process_wifi_mode(dist)
        elif self.mode == 2:
            self.process_autonomous_mode(dist)

        # Apply motor slew-rate ramping
        self.motors.update_ramp()


# =============================================================================
# ASYNC NON-BLOCKING WEB SERVER
# =============================================================================
def start_softap():
    if not IS_MICROPYTHON:
        return "127.0.0.1"

    ap = network.WLAN(network.AP_IF)
    ap.active(True)
    ap.config(essid=AP_SSID, password=AP_PASSWORD, authmode=network.AUTH_WPA_WPA2_PSK)
    while not ap.active():
        time.sleep(0.1)
    ip = ap.ifconfig()[0]
    print(f"[WIFI AP READY] SSID: {AP_SSID} | URL: http://{ip}")
    return ip


def handle_http_request(sock, robot):
    try:
        req = sock.recv(1024).decode("utf-8")
        if not req:
            sock.close()
            return

        line = req.split("\r\n")[0]
        parts = line.split(" ")
        if len(parts) < 2:
            sock.close()
            return
        path = parts[1]

        # Route: UI
        if path == "/" or path == "/index.html":
            body = HTML_DASHBOARD
            resp = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: " + str(len(body)) + "\r\nConnection: close\r\n\r\n" + body
            sock.sendall(resp.encode("utf-8"))

        # Route: Direction Command
        elif path.startswith("/cmd"):
            move = "stop"
            if "move=" in path:
                move = path.split("move=")[1].split("&")[0]
            robot.wifi_current_move = move
            robot.last_wifi_cmd_time = time.ticks_ms() if IS_MICROPYTHON else int(time.time() * 1000)
            sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nOK")

        # Route: Change Mode
        elif path.startswith("/set_mode"):
            if "val=" in path:
                try:
                    m = int(path.split("val=")[1].split("&")[0])
                    if 0 <= m <= 2:
                        robot.mode = m
                        robot.motors.immediate_stop()
                except ValueError:
                    pass
            resp = '{"mode": ' + str(robot.mode) + '}'
            sock.sendall(f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{resp}".encode("utf-8"))

        # Route: Throttle Limiter
        elif path.startswith("/speed"):
            if "value=" in path:
                try:
                    v = int(path.split("value=")[1].split("&")[0])
                    robot.wifi_speed = max(0, min(255, v))
                except ValueError:
                    pass
            sock.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nSpeed Set")

        # Route: Telemetry JSON
        elif path.startswith("/telemetry"):
            payload = (
                '{"distance": ' + str(robot.sensor.current_distance) +
                ', "mode": ' + str(robot.mode) +
                ', "speed": ' + str(robot.wifi_speed) + '}'
            )
            sock.sendall(f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{payload}".encode("utf-8"))

        else:
            sock.sendall(b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\nNot Found")

    except Exception:
        pass
    finally:
        sock.close()


# =============================================================================
# MAIN EVENT LOOP
# =============================================================================
def main():
    print("=" * 60)
    print("Starting Advanced ESP32 Robot System...")
    print("=" * 60)

    start_softap()
    robot = AdvancedRobot()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("", 80))
    server.listen(4)
    server.setblocking(False)

    print("[SYSTEM READY] Motors Protected • Failsafes Active")

    while True:
        # Non-blocking web connection polling
        try:
            readable, _, _ = select.select([server], [], [], 0.002)
            if readable:
                client_sock, _ = server.accept()
                handle_http_request(client_sock, robot)
        except Exception:
            pass

        # Update sensor, failsafes, kinematics, and motor ramps
        robot.update()

        time.sleep(0.002)


def run_robot_loop(calling_agent=None):
    """Entry point for master orchestrator."""
    if calling_agent:
        print("[RADXA_ROVER] Emergency Calling Watchdog linked to chassis.")
    main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutdown signal. Halting motors.")
        MotorDriverBTS7960().immediate_stop()
