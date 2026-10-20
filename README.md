import math
import re
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Room Digital Twin",
    page_icon="🏠",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

# 1 simulation cycle = 1 simulated minute
SIMULATION_STEP = 60.0

THERMAL_CAPACITY = 90_000.0       # J/K
UA_CLOSED = 40.0                  # W/K
WINDOW_UA_FACTOR = 2.5

HEATER_POWER = 2000.0             # W
LIGHT_POWER = 100.0               # W
LIGHT_K = 5.0                     # lux per dimmer %

DEFAULT_E_TARGET = 500.0          # lux

FROST_LIMIT = 7.0                 # °C
TEMP_DEADBAND = 0.5               # °C

OCCUPIED_SETPOINT = 21.0          # °C
VACANT_SETPOINT = 17.0             # °C

VACANCY_TIMEOUT = 300.0           # 5 minutes
HEATER_ANOMALY_TIME = 600.0       # 10 minutes
HEATER_MIN_RISE = 0.10            # °C

MODEL_TEMP_THRESHOLD = 0.50        # °C
MODEL_LIGHT_THRESHOLD = 50.0       # lux

TARIFF = 0.20                      # €/kWh
ALWAYS_ON_POWER = HEATER_POWER + LIGHT_POWER


# ============================================================
# PHYSICAL LAYER
# ============================================================

class PhysicalRoom:
    """Physical layer: room + sensors + actuators."""

    def __init__(self):
        # Sensors
        self.T_in = 20.0
        self.RH = 50.0
        self.T_out = 12.0
        self.E_day = 300.0
        self.E = 300.0
        self.occ = True
        self.P_el = 0.0

        # Actuators
        self.heater_on = False
        self.dimmer = 0.0
        self.window_open = False

        # Internal state
        self.vacancy_seconds = 0.0

        # Fault simulation
        self.heater_fault = False
        self.lighting_fault = False

    def reset(self):
        self.__init__()

    def update_vacancy(self, seconds):
        if self.occ:
            self.vacancy_seconds = 0.0
        else:
            self.vacancy_seconds += seconds

    def calculate_power(self):
        heater_power = HEATER_POWER if self.heater_on else 0.0

        if self.lighting_fault:
            light_power = LIGHT_POWER
        else:
            light_power = LIGHT_POWER * (self.dimmer / 100.0)

        self.P_el = heater_power + light_power
        return self.P_el


# ============================================================
# VIRTUAL LAYER - THERMAL MODEL
# ============================================================

class ThermalModel:
    def __init__(self):
        self.C = THERMAL_CAPACITY
        self.UA_closed = UA_CLOSED

    def effective_UA(self, window_open):
        return self.UA_closed * WINDOW_UA_FACTOR if window_open else self.UA_closed

    def time_constant(self, window_open=False):
        return self.C / self.effective_UA(window_open)

    def steady_state_temperature(
        self,
        T_out,
        heater_on,
        window_open=False,
    ):
        ua = self.effective_UA(window_open)
        heater_power = HEATER_POWER if heater_on else 0.0
        return T_out + heater_power / ua

    def predict(
        self,
        T0,
        T_out,
        heater_on,
        dt,
        window_open=False,
    ):
        ua = self.effective_UA(window_open)
        tau = self.C / ua
        heater_power = HEATER_POWER if heater_on else 0.0
        T_ss = T_out + heater_power / ua

        return T_ss - (T_ss - T0) * math.exp(-dt / tau)

    def time_to_setpoint(
        self,
        T0,
        T_out,
        heater_on,
        T_sp,
        window_open=False,
    ):
        ua = self.effective_UA(window_open)
        tau = self.C / ua
        heater_power = HEATER_POWER if heater_on else 0.0
        T_ss = T_out + heater_power / ua

        if heater_on and T_sp > T0 and T_sp < T_ss:
            numerator = T_ss - T_sp
            denominator = T_ss - T0

            if denominator <= 0:
                return None

            ratio = numerator / denominator

            if ratio <= 0 or ratio >= 1:
                return None

            return -tau * math.log(ratio)

        return None


# ============================================================
# VIRTUAL LAYER - LIGHTING MODEL
# ============================================================

class LightingModel:
    def __init__(self, k=LIGHT_K):
        self.k = k

    def predict(self, E_day, dimmer):
        dimmer = max(0.0, min(100.0, dimmer))
        return E_day + self.k * dimmer

    def required_dimmer(self, E_day, E_target):
        required = (E_target - E_day) / self.k
        return max(0.0, min(100.0, required))


# ============================================================
# VIRTUAL LAYER - ENERGY MODEL
# ============================================================

class EnergyModel:
    def __init__(self, tariff=TARIFF):
        self.energy_kwh = 0.0
        self.tariff = tariff

    def update(self, power_w, dt_seconds):
        self.energy_kwh += power_w * dt_seconds / 3_600_000.0

    def cost(self):
        return self.energy_kwh * self.tariff

    def savings_kwh(self, elapsed_seconds):
        baseline = ALWAYS_ON_POWER * elapsed_seconds / 3_600_000.0
        return max(0.0, baseline - self.energy_kwh)

    def savings_eur(self, elapsed_seconds):
        return self.savings_kwh(elapsed_seconds) * self.tariff


# ============================================================
# REASONING LAYER
# ============================================================

class ReasoningEngine:
    def __init__(self):
        self.messages = []
        self.anomalies = []

        # One-cycle manual overrides
        self.manual_heater_override = None
        self.manual_dimmer_override = None

        # Heater anomaly tracking
        self.heater_on_seconds = 0.0
        self.heater_on_start_temperature = None

    def determine_setpoint(self, room, current_hour):
        night = current_hour >= 23 or current_hour < 6

        if not room.occ or night:
            return VACANT_SETPOINT

        return OCCUPIED_SETPOINT

    def decide_heater(self, room, T_sp):
        self.messages = []
        T = room.T_in
        T_out = room.T_out

        # Rule 1: Frost protection
        if T < FROST_LIMIT:
            self.messages.append(
                "Frost protection: T_in < 7°C → heater ON."
            )
            return True

        # Rule 4: Free heat lockout
        if T_out >= T_sp:
            self.messages.append(
                "Free-heat lockout: T_out >= T_sp → heater OFF."
            )
            return False

        # Manual one-cycle override
        if self.manual_heater_override is not None:
            command = self.manual_heater_override
            self.messages.append(
                "Manual heater command → "
                + ("ON." if command else "OFF.")
            )
            return command

        # Rule 2: Hysteresis
        if T < T_sp - TEMP_DEADBAND:
            self.messages.append(
                f"Hysteresis: T_in < {T_sp - TEMP_DEADBAND:.1f}°C → heater ON."
            )
            return True

        if T > T_sp + TEMP_DEADBAND:
            self.messages.append(
                f"Hysteresis: T_in > {T_sp + TEMP_DEADBAND:.1f}°C → heater OFF."
            )
            return False

        self.messages.append(
            "Hysteresis deadband: maintaining previous heater state."
        )
        return room.heater_on

    def decide_lighting(self, room, E_target):
        # Rule 5: Vacancy shutoff
        if not room.occ and room.vacancy_seconds >= VACANCY_TIMEOUT:
            self.messages.append(
                "Vacancy timeout reached → dimmer 0%."
            )
            return 0.0

        # Rule 7: Daylight sufficient lockout
        if room.E_day >= E_target:
            self.messages.append(
                "Daylight sufficient → luminaire OFF."
            )
            return 0.0

        # Manual one-cycle override
        if self.manual_dimmer_override is not None:
            dimmer = self.manual_dimmer_override
            self.messages.append(
                f"Manual dimmer command → {dimmer:.0f}%."
            )
            return max(0.0, min(100.0, dimmer))

        # Rule 6: Daylight harvesting
        dimmer = (E_target - room.E_day) / LIGHT_K
        dimmer = max(0.0, min(100.0, dimmer))

        self.messages.append(
            f"Daylight harvesting → dimmer {dimmer:.0f}%."
        )
        return dimmer

    def update_heater_anomaly(self, room, dt):
        if room.heater_on:
            if self.heater_on_seconds == 0.0:
                self.heater_on_start_temperature = room.T_in

            self.heater_on_seconds += dt

            if (
                self.heater_on_seconds >= HEATER_ANOMALY_TIME
                and self.heater_on_start_temperature is not None
                and room.T_in - self.heater_on_start_temperature < HEATER_MIN_RISE
            ):
                message = (
                    "Heater anomaly: heater commanded ON for more than "
                    "10 minutes without sufficient temperature rise."
                )
                if message not in self.anomalies:
                    self.anomalies.append(message)
        else:
            self.heater_on_seconds = 0.0
            self.heater_on_start_temperature = None

    def detect_lighting_anomaly(self, room):
        if (
            room.lighting_fault
            and not room.occ
            and room.vacancy_seconds >= VACANCY_TIMEOUT
        ):
            message = (
                "Lighting anomaly: luminaire remains active "
                "while the room is vacant."
            )
            if message not in self.anomalies:
                self.anomalies.append(message)

    def clear_manual_overrides(self):
        self.manual_heater_override = None
        self.manual_dimmer_override = None


# ============================================================
# DIGITAL TWIN
# ============================================================

class DigitalTwin:
    def __init__(self):
        self.physical = PhysicalRoom()

        self.thermal_model = ThermalModel()
        self.lighting_model = LightingModel()
        self.energy_model = EnergyModel()
        self.reasoning = ReasoningEngine()

        # State variables
        self.T_sp = OCCUPIED_SETPOINT
        self.E_target = DEFAULT_E_TARGET
        self.duty_cycle = 0.0
        self.heater_runtime_seconds = 0.0
        self.predicted_time_to_setpoint = None

        # Residuals
        self.temperature_residual = 0.0
        self.illuminance_residual = 0.0

        # Latest predictions
        self.last_predicted_temperature = self.physical.T_in
        self.last_predicted_illuminance = self.physical.E

        self.health_status = "HEALTHY"

        self.simulation_seconds = 0.0
        self.history = []

        # Simulated clock starts at 12:00
        self.current_clock = datetime(2026, 1, 1, 12, 0, 0)

    def reset(self):
        self.__init__()

    def clock_string(self):
        return self.current_clock.strftime("%H:%M")

    def advance_clock(self):
        self.current_clock += timedelta(seconds=SIMULATION_STEP)

    def process_command(self, text):
        command = text.strip().lower()

        if not command:
            return "No command entered."

        # Window
        if re.search(r"\b(open|άνοιξε|ανοιξε)\b.*\b(window|παράθυρο|παραθυρο)\b", command):
            self.physical.window_open = True
            return "Window opened."

        if re.search(r"\b(close|κλείσε|κλεισε)\b.*\b(window|παράθυρο|παραθυρο)\b", command):
            self.physical.window_open = False
            return "Window closed."

        # Occupancy
        if re.search(r"\b(vacant|empty|απουσία|απουσια|άδειο|αδειο)\b", command):
            self.physical.occ = False
            return "Room set to vacant."

        if re.search(r"\b(occupied|presence|παρουσία|παρουσια|occupied)\b", command):
            self.physical.occ = True
            return "Room set to occupied."

        # Heater
        if re.search(r"\b(heater|θέρμανση|θερμανση)\b.*\b(on|ενεργοποίηση|άναψε|αναψε)\b", command):
            self.reasoning.manual_heater_override = True
            return "Manual heater ON command accepted for the next cycle."

        if re.search(r"\b(heater|θέρμανση|θερμανση)\b.*\b(off|απενεργοποίηση|σβήσε|σβησε)\b", command):
            self.reasoning.manual_heater_override = False
            return "Manual heater OFF command accepted for the next cycle."

        # Lights ON/OFF
        if re.search(r"\b(lights?|φώτα|φωτα|luminaire|φωτισμός|φωτισμος)\b.*\b(off|κλείσε|κλεισε|σβήσε|σβησε)\b", command):
            self.reasoning.manual_dimmer_override = 0.0
            return "Manual lights OFF command accepted for the next cycle."

        if re.search(r"\b(lights?|φώτα|φωτα|luminaire|φωτισμός|φωτισμος)\b.*\b(on|άναψε|αναψε|ενεργοποίηση)\b", command):
            self.reasoning.manual_dimmer_override = 100.0
            return "Manual lights ON command accepted for the next cycle."

        # Dimmer percentage
        dimmer_match = re.search(
            r"(?:dimmer|brightness|φωτισμός|φωτισμος|ένταση|ενταση)\s*(?:to)?\s*(\d+(?:\.\d+)?)\s*%?",
            command,
        )
        if dimmer_match:
            dimmer = max(0.0, min(100.0, float(dimmer_match.group(1))))
            self.reasoning.manual_dimmer_override = dimmer
            return f"Manual dimmer command accepted: {dimmer:.0f}%."

        # Outdoor temperature
        temp_match = re.search(
            r"(?:outside temperature|outdoor temperature|εξωτερική θερμοκρασία|εξωτερικη θερμοκρασια|outside|outdoor)\s*(-?\d+(?:\.\d+)?)",
            command,
        )
        if temp_match:
            self.physical.T_out = float(temp_match.group(1))
            return f"Outdoor temperature set to {self.physical.T_out:.1f}°C."

        # Queries
        if re.search(r"\b(temperature|θερμοκρασία|θερμοκρασια)\b", command):
            return f"Indoor temperature: {self.physical.T_in:.2f}°C."

        if re.search(r"\b(humidity|υγρασία|υγρασια)\b", command):
            return f"Relative humidity: {self.physical.RH:.1f}%."

        if re.search(r"\b(energy|ενέργεια|ενεργεια|κατανάλωση|καταναλωση)\b", command):
            return (
                f"Energy: {self.energy_model.energy_kwh:.4f} kWh, "
                f"cost: €{self.energy_model.cost():.4f}."
            )

        if re.search(r"\b(window|παράθυρο|παραθυρο)\b", command):
            return "Window: OPEN." if self.physical.window_open else "Window: CLOSED."

        if re.search(r"\b(occupancy|presence|παρουσία|παρουσια)\b", command):
            return "Room is occupied." if self.physical.occ else "Room is vacant."

        if re.search(r"\b(heater|θέρμανση|θερμανση)\b", command):
            return "Heater: ON." if self.physical.heater_on else "Heater: OFF."

        if re.search(r"\b(lights?|φώτα|φωτα|luminaire)\b", command):
            return f"Luminaire: {self.physical.dimmer:.0f}%."

        return (
            "Command not recognized. Examples: "
            "open window, close window, heater on, heater off, "
            "lights on, lights off, dimmer 50%, vacant, occupied."
        )

    def run_cycle(self):
        room = self.physical
        dt = SIMULATION_STEP

        # 1. Read sensors / update internal physical state
        room.update_vacancy(dt)

        # 2. Determine setpoint
        self.T_sp = self.reasoning.determine_setpoint(
            room,
            self.current_clock.hour,
        )

        # 3. Reasoning / actuator commands
        room.heater_on = self.reasoning.decide_heater(room, self.T_sp)
        room.dimmer = self.reasoning.decide_lighting(room, self.E_target)

        # 4. Virtual-layer one-step predictions
        predicted_temperature = self.thermal_model.predict(
            T0=room.T_in,
            T_out=room.T_out,
            heater_on=room.heater_on,
            dt=dt,
            window_open=room.window_open,
        )

        predicted_illuminance = self.lighting_model.predict(
            room.E_day,
            room.dimmer,
        )

        # 5. Physical evolution
        # Heater fault: command is ON, but no thermal effect is produced.
        actual_heater_on = room.heater_on and not room.heater_fault

        ua = self.thermal_model.effective_UA(room.window_open)
        heat_power = HEATER_POWER if actual_heater_on else 0.0

        dT_dt = (
            heat_power - ua * (room.T_in - room.T_out)
        ) / THERMAL_CAPACITY

        room.T_in += dT_dt * dt

        # Lighting fault: luminaire remains fully active.
        if room.lighting_fault:
            room.dimmer = 100.0

        room.E = room.E_day + LIGHT_K * room.dimmer
        room.E = max(0.0, room.E)

        # 6. Power and energy
        power = room.calculate_power()
        self.energy_model.update(power, dt)

        # 7. Anomaly detection
        self.reasoning.update_heater_anomaly(room, dt)
        self.reasoning.detect_lighting_anomaly(room)

        # 8. Model residuals
        self.temperature_residual = room.T_in - predicted_temperature
        self.illuminance_residual = room.E - predicted_illuminance

        if abs(self.temperature_residual) > MODEL_TEMP_THRESHOLD:
            message = (
                "Thermal model divergence: "
                f"residual={self.temperature_residual:.2f}°C"
            )
            if message not in self.reasoning.anomalies:
                self.reasoning.anomalies.append(message)

        if abs(self.illuminance_residual) > MODEL_LIGHT_THRESHOLD:
            message = (
                "Lighting model divergence: "
                f"residual={self.illuminance_residual:.1f} lux"
            )
            if message not in self.reasoning.anomalies:
                self.reasoning.anomalies.append(message)

        # 9. Time to setpoint
        self.predicted_time_to_setpoint = self.thermal_model.time_to_setpoint(
            T0=room.T_in,
            T_out=room.T_out,
            heater_on=room.heater_on,
            T_sp=self.T_sp,
            window_open=room.window_open,
        )

        # 10. Duty cycle
        if room.heater_on:
            self.heater_runtime_seconds += dt

        total_time = self.simulation_seconds + dt
        if total_time > 0:
            self.duty_cycle = (
                self.heater_runtime_seconds / total_time
            ) * 100.0

        # 11. Health status
        self.health_status = (
            "ANOMALY" if self.reasoning.anomalies else "HEALTHY"
        )

        # 12. Save predictions
        self.last_predicted_temperature = predicted_temperature
        self.last_predicted_illuminance = predicted_illuminance

        # 13. Advance simulation time
        self.simulation_seconds += dt
        self.advance_clock()

        # 14. Save complete history
        self.history.append(
            {
                "Time": self.clock_string(),
                "Simulated min": self.simulation_seconds / 60.0,
                "T_in": room.T_in,
                "T_out": room.T_out,
                "T_sp": self.T_sp,
                "T_predicted": predicted_temperature,
                "Temp residual": self.temperature_residual,
                "RH": room.RH,
                "E_day": room.E_day,
                "E": room.E,
                "E_target": self.E_target,
                "E_predicted": predicted_illuminance,
                "Light residual": self.illuminance_residual,
                "Occupancy": room.occ,
                "Heater": room.heater_on,
                "Dimmer": room.dimmer,
                "Window": room.window_open,
                "Power W": room.P_el,
                "Energy kWh": self.energy_model.energy_kwh,
                "Health": self.health_status,
            }
        )

        # One-cycle manual commands expire here.
        self.reasoning.clear_manual_overrides()

    def run_cycles(self, number):
        for _ in range(number):
            self.run_cycle()


# ============================================================
# SESSION STATE
# ============================================================

def session_state_is_valid(twin):
    required_physical = [
        "T_in", "RH", "T_out", "E_day", "E", "occ", "P_el",
        "heater_on", "dimmer", "window_open", "vacancy_seconds",
        "heater_fault", "lighting_fault",
    ]

    required_twin = [
        "thermal_model", "lighting_model", "energy_model",
        "reasoning", "simulation_seconds", "history",
    ]

    if twin is None or not hasattr(twin, "physical"):
        return False

    if not all(hasattr(twin.physical, attr) for attr in required_physical):
        return False

    if not all(hasattr(twin, attr) for attr in required_twin):
        return False

    return True


if "twin" not in st.session_state:
    st.session_state.twin = DigitalTwin()
elif not session_state_is_valid(st.session_state.twin):
    st.session_state.twin = DigitalTwin()

if "command_response" not in st.session_state:
    st.session_state.command_response = ""

twin = st.session_state.twin


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🌡️ Physical Layer")

    twin.physical.T_out = st.slider(
        "Outdoor temperature °C",
        min_value=-10.0,
        max_value=40.0,
        value=float(twin.physical.T_out),
        step=0.5,
    )

    twin.physical.RH = st.slider(
        "Relative humidity %",
        min_value=20.0,
        max_value=80.0,
        value=float(twin.physical.RH),
        step=1.0,
    )

    twin.physical.E_day = st.slider(
        "Daylight lux",
        min_value=0.0,
        max_value=2000.0,
        value=float(twin.physical.E_day),
        step=10.0,
    )

    twin.physical.occ = st.checkbox(
        "Occupancy / Presence",
        value=bool(twin.physical.occ),
    )

    st.divider()

    st.header("🎯 Virtual Targets")

    twin.E_target = st.slider(
        "Target illuminance lux",
        min_value=100.0,
        max_value=1000.0,
        value=float(twin.E_target),
        step=10.0,
    )

    st.divider()

    st.header("🧪 Fault Simulation")

    twin.physical.heater_fault = st.checkbox(
        "Simulate heater fault",
        value=bool(twin.physical.heater_fault),
    )

    twin.physical.lighting_fault = st.checkbox(
        "Simulate lighting fault",
        value=bool(twin.physical.lighting_fault),
    )


# ============================================================
# MAIN PAGE
# ============================================================

st.title("🏠 Smart Room Digital Twin")

st.caption(
    "Digital Twin architecture: "
    "Physical Layer → Virtual Layer → "
    "Reasoning Layer → Actuators → Feedback"
)


# ============================================================
# DIGITAL TWIN CONTROLS
# ============================================================

st.subheader("🎛️ Digital Twin Controls")

st.info(
    "1 cycle = 1 simulated minute. "
    "The Digital Twin executes the closed loop automatically; "
    "you do not need to enter a command for every cycle."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("▶️ Run 1 Cycle", use_container_width=True):
        twin.run_cycles(1)
        st.rerun()

with col2:
    if st.button("⏱️ Run 1 Minute", use_container_width=True):
        twin.run_cycles(1)
        st.rerun()

with col3:
    if st.button("⏱️ Run 10 Minutes", use_container_width=True):
        twin.run_cycles(10)
        st.rerun()

with col4:
    if st.button("⏱️ Run 60 Minutes", use_container_width=True):
        twin.run_cycles(60)
        st.rerun()

col5, col6 = st.columns(2)

with col5:
    if st.button("🔁 Run 60 Cycles", use_container_width=True):
        twin.run_cycles(60)
        st.rerun()

with col6:
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.twin = DigitalTwin()
        st.session_state.command_response = ""
        st.rerun()

st.divider()


# ============================================================
# NATURAL LANGUAGE COMMAND
# ============================================================

st.subheader("💬 Natural-Language Command")

command = st.text_input(
    "Enter a command",
    placeholder=(
        "e.g. open window, heater on, lights off, "
        "dimmer 50%, vacant"
    ),
)

if st.button("Send Command", use_container_width=False):
    if command.strip():
        response = twin.process_command(command)
        st.session_state.command_response = response

        # Run one closed-loop cycle after the external intervention.
        twin.run_cycle()
        st.rerun()

if st.session_state.command_response:
    st.success(st.session_state.command_response)


# ============================================================
# CURRENT STATE
# ============================================================

st.subheader("📊 Current State")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Indoor Temperature",
        f"{twin.physical.T_in:.2f} °C",
    )

with m2:
    st.metric(
        "Humidity",
        f"{twin.physical.RH:.1f} %",
    )

with m3:
    st.metric(
        "Illuminance",
        f"{twin.physical.E:.0f} lux",
    )

with m4:
    st.metric(
        "Electrical Power",
        f"{twin.physical.P_el:.0f} W",
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.subheader("🟢 System Status")

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.metric("Health", twin.health_status)

with s2:
    st.metric(
        "Simulation Time",
        f"{twin.simulation_seconds / 60:.0f} min",
    )

with s3:
    st.metric("Simulated Clock", twin.clock_string())

with s4:
    st.metric(
        "Target Temperature",
        f"{twin.T_sp:.1f} °C",
    )


# ============================================================
# ACTUATORS
# ============================================================

st.subheader("⚙️ Actuators")

a1, a2, a3 = st.columns(3)

with a1:
    st.metric(
        "Heater",
        "ON" if twin.physical.heater_on else "OFF",
    )

with a2:
    st.metric(
        "Luminaire",
        f"{twin.physical.dimmer:.0f} %",
    )

with a3:
    st.metric(
        "Window",
        "OPEN" if twin.physical.window_open else "CLOSED",
    )


# ============================================================
# REASONING LAYER
# ============================================================

st.subheader("🧠 Reasoning Layer")

if twin.reasoning.messages:
    for message in twin.reasoning.messages:
        st.write("•", message)
else:
    st.write("No reasoning messages yet. Run a simulation cycle.")


# ============================================================
# VIRTUAL LAYER
# ============================================================

st.subheader("🧮 Virtual Layer")

v1, v2, v3 = st.columns(3)

with v1:
    st.metric(
        "Predicted Temperature",
        f"{twin.last_predicted_temperature:.2f} °C",
    )

with v2:
    st.metric(
        "Predicted Illuminance",
        f"{twin.last_predicted_illuminance:.0f} lux",
    )

with v3:
    if twin.predicted_time_to_setpoint is None:
        value = "N/A"
    else:
        value = f"{twin.predicted_time_to_setpoint / 60:.1f} min"

    st.metric("Predicted Time to Setpoint", value)


# ============================================================
# MODEL PREDICTION AND RESIDUALS - CURRENT CYCLE
# ============================================================

st.subheader("📐 Model Prediction and Residuals")

r1, r2 = st.columns(2)

with r1:
    st.metric(
        "Temperature Residual",
        f"{twin.temperature_residual:.3f} °C",
    )

with r2:
    st.metric(
        "Illuminance Residual",
        f"{twin.illuminance_residual:.1f} lux",
    )

st.caption(
    "Residual = actual simulated value − model predicted value."
)


# ============================================================
# ENERGY
# ============================================================

st.subheader("⚡ Energy and Cost")

e1, e2, e3, e4 = st.columns(4)

with e1:
    st.metric(
        "Energy",
        f"{twin.energy_model.energy_kwh:.4f} kWh",
    )

with e2:
    st.metric(
        "Cost",
        f"€{twin.energy_model.cost():.4f}",
    )

with e3:
    st.metric(
        "Theoretical Always-On Savings",
        f"{twin.energy_model.savings_kwh(twin.simulation_seconds):.4f} kWh",
    )

with e4:
    st.metric(
        "Heater Duty Cycle",
        f"{twin.duty_cycle:.1f} %",
    )


# ============================================================
# ANOMALIES
# ============================================================

st.subheader("🚨 Anomalies")

if twin.reasoning.anomalies:
    for anomaly in twin.reasoning.anomalies:
        st.error(anomaly)
else:
    st.success("No anomalies detected.")


# ============================================================
# HISTORY
# ============================================================

st.subheader("📈 Simulation History")

if twin.history:
    df = pd.DataFrame(twin.history)

    st.dataframe(
        df.tail(20),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # 5.5.1 Temperature history
    # --------------------------------------------------------

    st.markdown("### Temperature History")

    st.line_chart(
        df.set_index("Simulated min")[
            [
                "T_in",
                "T_out",
                "T_sp",
            ]
        ]
    )

    # --------------------------------------------------------
    # 5.4 Prediction vs Actual - Temperature
    # --------------------------------------------------------

    st.markdown("### Prediction vs Actual – Temperature")

    st.line_chart(
        df.set_index("Simulated min")[
            [
                "T_in",
                "T_predicted",
            ]
        ]
    )

    st.markdown("### Temperature Residual")

    st.line_chart(
        df.set_index("Simulated min")[
            [
                "Temp residual",
            ]
        ]
    )

    # --------------------------------------------------------
    # 5.5.2 Humidity
    # --------------------------------------------------------

    st.markdown("### Humidity History")

    st.line_chart(
        df.set_index("Simulated min")[
            [
                "RH",
            ]
        ]
    )

    # --------------------------------------------------------
    # 5.5.3 Lighting history
    # --------------------------------------------------------

    st.markdown("### Illuminance History")

    st.line_chart(
        df.set_index("Simulated min")[
            [
                "E",
                "E_day",
                "E_target",
            ]
        ]
    )

    # --------------------------------------------------------
    # 5.4 Prediction vs Actual - Illuminance
    # --------------------------------------------------------

    st.markdown("### Prediction vs Actual – Illuminance")

    st.line_chart(
        df.set_index("Simulated min")[
            [
                "E",
                "E_predicted",
            ]
        ]
    )

    st.markdown("### Illuminance Residual")

    st.line_chart(
        df.set_index("Simulated min")[
            [
                "Light residual",
            ]
        ]
    )

    # --------------------------------------------------------
    # 5.5.4 Energy
    # --------------------------------------------------------

    st.markdown("### Energy and Power History")

    st.line_chart(
        df.set_index("Simulated min")[
            [
                "Power W",
                "Energy kWh",
            ]
        ]
    )

else:
    st.info(
        "No simulation history yet. Run one or more simulation cycles."
    )


# ============================================================
# FORMAL DIGITAL TWIN DESCRIPTION
# ============================================================

st.subheader("📘 Formal Digital Twin Representation")

st.markdown(
    """
The implemented Smart Room Digital Twin can be represented by the tuple:

**(P, A, X, R, M)**

- **P – Monitored parameters:** indoor temperature, relative humidity,
  outdoor temperature, daylight, illuminance, occupancy and electrical power.
- **A – Actuators:** heater, dimmable luminaire and window.
- **X – State variables:** temperature setpoint, illuminance target,
  actuator states, energy consumption, heater duty cycle,
  predicted time to setpoint and health/anomaly status.
- **R – Rules:** frost protection, hysteresis, setback, free-heat lockout,
  vacancy shutoff and daylight harvesting.
- **M – Models:** thermal model, lighting model and energy model.
"""
)


# ============================================================
# CLOSED-LOOP ARCHITECTURE
# ============================================================

st.subheader("🔄 Closed-Loop Architecture")

st.markdown(
    """
```text
┌──────────────────────────────┐
│       PHYSICAL LAYER         │
│                              │
│   Room + Sensors + Actuators │
└──────────────┬───────────────┘
               │ measurements
               ▼
┌──────────────────────────────┐
│        VIRTUAL LAYER         │
│                              │
│   Thermal Model              │
│   Lighting Model             │
│   Energy Model               │
└──────────────┬───────────────┘
               │ prediction
               ▼
┌──────────────────────────────┐
│       REASONING LAYER        │
│                              │
│   Rules + Anomaly Detection  │
└──────────────┬───────────────┘
               │ actuator commands
               ▼
┌──────────────────────────────┐
│      PHYSICAL ACTUATORS      │
│                              │
│   Heater / Lights / Window   │
└──────────────┬───────────────┘
               │ new physical state
               ▼
          ┌──────────┐
          │ FEEDBACK │
          └────┬─────┘
               │
               └──────────────► next cycle
"""
)

st.subheader("⏱️ Simulation Information")

st.write("**Cycle duration:** 60 seconds (1 simulated minute)")
st.write(
    f"**Current simulated time:** "
    f"{twin.simulation_seconds / 60:.0f} minutes"
)
st.write(
    "**Digital Twin loop:** Sensors → Model → Reasoning → "
    "Actuators → Physical evolution → Feedback"
)
st.write("**Thermal parameters:** C = 90 kJ/K, UA = 40 W/K")
st.write("**Heater power:** 2000 W")
st.write("**Luminaire power:** 100 W")
st.write("**Energy tariff:** €0.20/kWh")
