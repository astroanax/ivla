
![iros](../assets/_static/video/iros_challenge.gif "iros")

# InternManip: IROS 2025 Grand Challenge Toolkits


![track-badge](https://img.shields.io/badge/Track_1-Manipulation-blueviolet)

> This repository provides the **official baseline and evaluation toolkit** for
> **Track: Vision-Language Manipulation in Open Tabletop Environments**,
> featured at the *[IROS 2025 Workshop](https://internrobotics.shlab.org.cn/workshop/2025/)*.


Welcome to the **IROS 2025 Challenge of Multimodal Robot Learning in InternUtopia and Real World**!

## 🚀 Challenge Overview

In this challenge, participants will develop end-to-end policies that fuse vision and language to control robots in simulated physics-based environment.
Models are trained using the **InternManip** framework and **GenManip** dataset, and evaluated in a closed-loop benchmark on unseen private scenes.

This repository serves as the **starter kit and evaluation toolkit**—you can use it to:
- Implement your own policy models
- Train them on GenManip public data
- Submit them via Docker for final evaluation


## 📚 Table of Contents

- [🚀 Challenge Overview](#-challenge-overview)
- [🛠️ Codebase & Tools](#️-codebase--tools)
- [📌 Task Definition](#-task-definition)
- [🔍 Evaluation Protocol](#-evaluation-protocol)
- [🧠 Key Challenges](#-key-challenges)
- [💻 Resource Configuration](#-resource-configuration)
- [📆 Timeline](#-timeline)
- [🏆 Prizes](#-prizes)
- [✅ Submission Guidelines](#-submission-guidelines)
- [📎 Rules & Eligibility](#-rules--eligibility)
- [📬 Contact & Support](#-contact--support)



## 🛠️ Codebase & Tools

- **Starter Code**: [InternManip GitHub Repo](https://github.com/InternRobotics/InternManip)
- **Data Format**: GenManip public benchmark
- **Simulation & Evaluation**: InternUtopia physics-based simulator
- **Docker**: A base image with all dependencies and starter scripts will be provided


## 📌 Task Definition

Participants must:
1. **Implement a custom control policy** using `InternManip`, compatible with the GenManip data format
2. **Train on GenManip public data** using supported loaders and APIs
3. **Submit a Docker image** that runs closed-loop rollouts on our private test set
4. Compete for **highest success rate** on the held-out test scenes



## 🔍 Evaluation Protocol

Submissions will be automatically evaluated on three splits:
- `val_seen`
- `val_unseen`
- `test` (private)

Evaluation metrics:
- ✅ **Success Rate**: average success per episode




## 🧠 Key Challenges


- **Multimodal Fusion**: Effectively combining visual observations and language instructions to drive a unified perception–decision–control pipeline.

- **Instruction Understanding & Skill Execution**: Interpreting diverse natural language instructions and executing corresponding multi-skill behaviors using robotic arms or mobile manipulators in physics-based environments.

- **Task & Object Generalization**: Ensuring generalization across tasks, objects, and spatial layouts to enable long-horizon manipulation in open, dynamic tabletop scenes.




## 💻 Resource Configuration

Each submission runs on a dedicated evaluation machine with:
- **CPU**: 14 cores
- **RAM**: 100 GB
- **GPU**: 1 × NVIDIA GeForce RTX 4090


## 📆 Timeline

| Phase               | Date & Time (CST)              |
|--------------------|-------------------------------|
| Test Phase Starts   | Aug 5, 2025 @ 00:00 CST        |
| Submission Deadline | Sep 30, 2025 @ 23:59 CST       |

Submission rules:
- ⏱ **Max Submissions**: 5 per day / 50 per month / 50 total
- 🚫 **Concurrent Submissions**: 1 at a time



## 🏆 Prizes

| Rank       | Prize                            |
|------------|----------------------------------|
| 🥇 1st      | \$10,000 + \$1,500 travel + cert |
| 🥈 2nd      | \$5,000 + \$1,500 travel + cert  |
| 🥉 3rd      | \$3,000 + \$1,500 travel + cert  |
| 4th–10th    | Certificate & finalist awards    |



## ✅ Submission Guidelines

- Teams must implement a `custom_policy()` class (see [GUIDELINES.md](./guidelines.md))
- Docker image must include all dependencies for inference
- Final Dockerfile and inference code must be open-sourced
- Public datasets and pretrained models are allowed
- **🚨 Strictly no access to test scenes or results outside official platform**



## 📎 Rules & Eligibility

- 👥 Team size: up to 10 members
- 🧪 One submission account per team
- 🔒 No access to private test data
- 🕑 Late or invalid submissions will be disqualified



## 📬 Contact & Support

- **Discord**: Join via invite link in the repo
- **WeChat**: Scan QR code in the repo
- **Email**: [embodiedai@pjlab.org.cn](mailto:embodiedai@pjlab.org.cn)



> 😄 Good luck, and we look forward to your innovations!
