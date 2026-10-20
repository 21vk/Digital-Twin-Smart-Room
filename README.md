# Smart Room Digital Twin

## 1. Project Overview

The Smart Room Digital Twin is a Python-based simulation application developed using Streamlit. It represents the state of a smart room and demonstrates how a digital twin can monitor simulated environmental conditions, estimate energy consumption, and apply automated control rules.

## 2. Main Features

- Simulation of indoor temperature, humidity and illuminance.
- Thermal and lighting models.
- Energy consumption and cost calculations.
- Automated control rules for heating and lighting.
- Manual commands for interacting with the simulated room.
- Anomaly detection and model prediction residuals.
- Dashboard with simulation status, history and results.

## 3. System Architecture

The application is organized into three main layers:

- **Physical Layer:** Represents the simulated room state and actuators.
- **Virtual Layer:** Models room behavior and predicts environmental conditions.
- **Reasoning Layer:** Applies control rules and evaluates the simulated system state.

The application uses simulated data and does not require physical sensors or actuators.

## 4. Technologies

- Python
- Streamlit
- Simulation-based thermal and lighting models

## 5. Running the Application

The application is intended to run locally on a computer with Python installed.

The exact installation requirements and execution command will be provided after verifying the application source file and its dependencies.

## 6. Example Commands

The application supports example commands such as:

- `open window`
- `heater on`
- `lights off`
- `dimmer 50%`
- `vacant`

These commands demonstrate manual interaction with the simulated room.

## 7. Demonstration

The application can be used to execute simulation cycles, observe changes in the room state, and examine energy calculations, control decisions and simulation history.

## 8. Project Purpose

The purpose of this project is to demonstrate the basic concepts of a Smart Room Digital Twin through a software-based simulation and an interactive dashboard.
