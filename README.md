# OPEN ARM Robotv1 Development Course
<img width="497" height="802" alt="image" src="https://github.com/user-attachments/assets/4185ec92-c0fb-43ea-8e65-98405d13fb2c" />

# Workflow
                 PHASE 1 — SIMULATION
                        │
                        ▼
             OpenArm V1 MuJoCo
                        │
                 cube + table
                        │
                        ▼
             Teleoperation / scripted
              successful demonstrations
                        │
                        ▼
              LeRobot V3 dataset
                        │
                        ▼
                 ACT training
                        │
                        ▼
             ACT policy checkpoint
                        │
                        ▼
             ┌──────────────────┐
             │ Sim evaluation   │
             │ 100 episodes     │
             └──────────────────┘
                        │
                  ≥ good success
                        │
                        ▼
             PHASE 2 — REAL ROBOT
                        │
                 OpenArm V1
                        │
                 RGB camera
                        │
                        ▼
              same observation
                        │
                        ▼
                  ACT policy
                        │
                        ▼
                 ROS2 controller
                        │
                        ▼
                  OpenArm V1

