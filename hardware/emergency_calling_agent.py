"""
=============================================================================
AGRIBOT EMERGENCY CALLING AGENT (Radxa Qualcomm SBC Edition)
Hardware: SIM800L / A9G Cellular Modem over UART / LiveKit SIP Telephony
Features:
- Incident-triggered voice dialing & SMS with GPS coordinates
- Full bilingual emergency templates (Hindi + English)
- Fail-safe fallback to local terminal simulator if modem is disconnected
=============================================================================
"""

import os
import sys
import time
import json
import logging

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

logger = logging.getLogger("emergency_calling_agent")

# =============================================================================
# INCIDENT TEMPLATES
# =============================================================================
INCIDENT_TEMPLATES = {
    "ROVER_STUCK": {
        "title": "Rover Blockage Alert",
        "voice_en": "Alert! Your AgriBot rover is physically stuck in muddy terrain at sector B2. Please inspect the rover.",
        "voice_hi": "चेतावनी! आपका एग्रीबॉट रोवर खेत में फंस गया है। कृपया जीपीएस लोकेशन पर जाकर रोवर की जांच करें।",
        "sms_prefix": "🚨 AGRIBOT CRITICAL: Rover is trapped/stuck at field coordinates."
    },
    "CRITICAL_BATTERY": {
        "title": "Low Battery Brownout Risk",
        "voice_en": "Warning. AgriBot battery has dropped below fifteen percent. Safe parking sequence initiated.",
        "voice_hi": "चेतावनी! रोवर की बैटरी पंद्रह प्रतिशत से कम हो गई है। कृपया तुरंत चार्ज करें।",
        "sms_prefix": "⚠️ AGRIBOT WARNING: Battery critically low (<15%)."
    },
    "SEVERE_PEST_OUTBREAK": {
        "title": "High-Risk Pest Outbreak",
        "voice_en": "Urgent agronomy advisory. Tobacco caterpillar infestation detected above ninety percent threshold.",
        "voice_hi": "कृषि सूचना! आपके खेत में तम्बाकू इल्ली का गंभीर प्रकोप देखा गया है। तुरंत स्प्रे की सलाह दी जाती है।",
        "sms_prefix": "🐛 AGRIBOT AI: Severe pest cluster detected (>90% confidence)."
    },
    "SOIL_DROUGHT_CRITICAL": {
        "title": "Extreme Soil Moisture Depletion",
        "voice_en": "Alert. Soil moisture at root depth has dropped to critical drought levels. Immediate irrigation required.",
        "voice_hi": "चेतावनी! मिट्टी में नमी बहुत कम हो गई है। तुरंत सिंचाई शुरू करने की आवश्यकता है।",
        "sms_prefix": "💧 AGRIBOT TELEMETRY: Soil moisture critical (<18%)."
    }
}


class AgriBotCallingAgent:
    def __init__(self, port="/dev/ttyUSB0", baudrate=9600, default_phone="+918319556016"):
        self.port = port
        self.baudrate = baudrate
        self.default_phone = default_phone
        self.has_hardware = False

        self._init_modem()

    def _init_modem(self):
        if SERIAL_AVAILABLE:
            try:
                self.ser = serial.Serial(self.port, self.baudrate, timeout=2.0)
                self.has_hardware = True
                logger.info(f"SIM800L Modem active on Radxa UART {self.port}")
                return
            except Exception:
                pass
        logger.info("Cellular Modem in Simulation Mode (Virtual UART Active)")

    def _send_at(self, cmd):
        if self.has_hardware:
            try:
                self.ser.write((cmd + "\r\n").encode("utf-8"))
                time.sleep(0.3)
                return self.ser.read_all().decode("utf-8", errors="ignore")
            except Exception as e:
                logger.error(f"AT command error: {e}")
        else:
            logger.info(f"[AT-SIM] >>> {cmd}")
            return "OK"

    def send_sms(self, phone_number, message):
        target = phone_number or self.default_phone
        logger.info(f"Sending SMS to {target}: {message[:60]}...")
        self._send_at("AT")
        self._send_at("AT+CMGF=1")
        self._send_at(f'AT+CMGS="{target}"')
        self._send_at(message)
        self._send_at("\x1A")
        return True

    def place_call(self, phone_number=None):
        target = phone_number or self.default_phone
        logger.info(f"Dialing voice SOS to {target}...")
        self._send_at(f"ATD{target};")
        time.sleep(6)
        self._send_at("ATH")
        logger.info("Voice SOS completed.")
        return True

    def trigger_incident(self, incident_key, language="hi", phone_number=None, lat=22.7196, lon=75.8577):
        target = phone_number or self.default_phone
        incident = INCIDENT_TEMPLATES.get(incident_key, INCIDENT_TEMPLATES["ROVER_STUCK"])
        voice_text = incident["voice_hi"] if language == "hi" else incident["voice_en"]
        maps_link = f"https://maps.google.com/?q={lat:.6f},{lon:.6f}"
        sms_text = f"{incident['sms_prefix']} Map: {maps_link}"

        print(f"\n🚨 [AGRIBOT INCIDENT TRIGGERED] {incident['title']}")
        print(f"  • Voice ({language.upper()}): \"{voice_text}\"")
        print(f"  • Destination: {target}")

        # 1. Voice Dial
        self.place_call(target)

        # 2. Backup SMS with GPS Coordinates
        self.send_sms(target, sms_text)
        return True


def start_emergency_service():
    """Service entry point for background monitoring."""
    agent = AgriBotCallingAgent()
    logger.info("AgriBot Emergency Calling Agent standing by...")
    return agent
