# Sensor plan: comparing the sim with the running system

Logging: ESP boards → Home Assistant → the user's server (small web page). Already in place:
reservoir pH, EC and TDS, plus 1-Wire thermometers. Planned: an overhead high-definition
camera of the greenhouse and a thermal camera per table. The entity list and the history
export format (Home Assistant REST/history API, CSV or MQTT) come later; the importer is
built against them.

## Constraints

- **Gas sensor range.** DFRobot's electrochemical gas sensors are low range (O₃ 0–10 ppm,
  NO₂ 0–20 ppm). The sim predicts the drum vent at 10²–10⁴ ppm. Mount them **after** the
  MnO₂ destruct (breakthrough, destruct efficiency) and in the room (safety). Raw reactor
  gas would need a high-range analyser or a diluted sample line; the water chemistry
  answers that question instead.
- **No DFRobot nitrate probe.** Use nitrate strips per batch, or a nitrate ISE (Hanna,
  Vernier). Nitrate per kWh is the single most useful plasma calibration.
- **Plasma power.** DFRobot's wattmeters can't take the 120 V bus. Log the supply's AC
  input with an energy-metering smart plug in HA, plus an occasional Lissajous measurement
  (100 nF in the ground strap, 1000:1 probe) for the true discharge power.

## PAW skid: calibrates the plasma model

| Sensor | Part | Where | Calibrates |
| --- | --- | --- | --- |
| Ozone, electrochemical, I2C, 0–10 ppm | DFRobot SEN0321 | after destruct; room | O₃ yield, destruct efficiency, exposure alarm |
| NO₂, factory calibrated, 0–20 ppm | DFRobot SEN0471 | after stage 2 / destruct; room | NOx slip, safety (STEL 1 ppm) |
| ORP | DFRobot SEN0165 | drum | dissolved ozone, hold before dosing |
| pH, EC | existing res probes (or a second set) | drum | batch pH vs time: the main MJ/mol fit |
| IR thermometer (MLX90614) | DFRobot (check SKU) | at the bare alumina above the top cap, outside the HV clearance | alumina temperature (model: ~257 °C) |
| Temperature | 1-Wire | drum water, pipe exterior, HV enclosure | module heat balance, drum heating |
| AC energy | smart plug | supply input | plasma kWh per batch |
| Nitrate | strips / ISE | drum, per batch | fixed N per kWh |

Reading the vent: ozone near zero with NO₂ present is the hot-gap NOx regime.

## NFT: validates the water and climate models

| Sensor | Part | Validates |
| --- | --- | --- |
| Dissolved oxygen | DFRobot SEN0237-A | reservoir / outlet DO |
| Air T + RH (SHT40) | DFRobot SEN0428, one per table | evaporation, transpiration (VPD) |
| CO₂ (SCD41, + T/RH) | DFRobot SEN0536 | pH model CO₂, plants, room leak check |
| Water flow, ½″ | DFRobot SEN0217 or SEN0550 | hydraulic solve (pump curve, channel flows) |
| Reservoir level (waterproof ultrasonic) | — | daily water use: transpiration + evaporation + top-off |
| 1-Wire T at a channel inlet and outlet per bank | existing | channel temperature rise, mylar vs bare |
| Chiller and pump energy | smart plugs | chiller duty and kWh |
| PAR (quantum sensor) or lux | non-DFRobot / lux proxy | radiant load input |
| Overhead camera | planned | canopy area → leaf area → transpiration and growth |
| Thermal camera per table | planned | leaf and channel temperatures |

## Safety (independent of the sim)

- Water-leak sensor under the drum and the HV enclosure.
- Room O₃ and NO₂ wired to an HA alert that also cuts the plasma.

## Records the replay needs

A batch log: start and stop times, strip readings, and manual changes (air flow, burst
duty, top-offs). The sim replays a batch from these events.
