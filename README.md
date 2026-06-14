# 🍄 Super Mario Bros — Double Deep Q-Network (DDQN)

An end-to-end Reinforcement Learning agent trained to play **Super Mario Bros (World 1-1)** from raw pixels using a **Double Deep Q-Network (DDQN)** built in PyTorch.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ncKQQKUFx0KtZuFt3xcW9PkEu8JlVjmf?usp=sharing)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---
<img width="1164" height="1164" alt="image" src="https://github.com/user-attachments/assets/079eebb3-bf4a-4ff5-89f1-deeb6d678dc5" />
<img width="521" height="490" alt="image" src="https://github.com/user-attachments/assets/540461ed-0f51-4e72-839b-6b5817babcaa" />



## 🎯 Project Overview

This project implements a DDQN agent that learns to play Super Mario Bros entirely from visual input — no hand-crafted rules, no access to internal game state. The agent observes stacked grayscale frames and outputs joystick actions through a convolutional neural network, improving its strategy over hundreds of training episodes via trial, error, and reward signals.

---

## 🧠 Network Architecture

The CNN takes 4 stacked grayscale frames (84×84) as input and outputs a Q-value for each of 7 possible actions.

| Layer | Details |
|---|---|
| Input | 4 × 84 × 84 stacked grayscale frames |
| Conv1 | 32 filters, kernel 8×8, stride 4, ReLU |
| Conv2 | 64 filters, kernel 4×4, stride 2, ReLU |
| Conv3 | 64 filters, kernel 3×3, stride 1, ReLU |
| FC1 | 3136 → 512, ReLU |
| Output | 512 → 7 Q-values (one per action) |

---

## ⚙️ Key Design Decisions

| Component | Choice | Reason |
|---|---|---|
| Algorithm | Double DQN | Decouples action selection and evaluation — reduces Q overestimation vs vanilla DQN |
| Loss Function | Huber (Smooth L1) | Robust to large TD-error spikes that occur on death events |
| Reward Shaping | Clip to [-15, 15] → normalize to [-1, 1] | Preserves magnitude signal while keeping gradients bounded |
| Epsilon Decay | 1.0 → 0.02 over 30,000 steps | Exploration collapses well within the 600-episode training budget |
| Frame Stacking | 4 frames | Gives the agent a sense of velocity and motion direction |
| Replay Buffer | 30,000 transitions | Breaks temporal correlation between training samples |
| Target Network Sync | Every 1,000 steps | Stabilizes Bellman targets throughout training |
| Gradient Clipping | max_norm = 10.0 | Prevents runaway weight updates during large loss spikes |
| Checkpointing | Every 50 episodes | Ensures progress is never lost due to Colab session timeouts |

---

## 🔑 Why Double DQN?

Standard DQN uses the same network to both **select** and **evaluate** the next action — this causes systematic overestimation of Q-values. DDQN decouples the two responsibilities:

- **Standard DQN** → `target = r + γ · max_a Q_target(s', a)` ← prone to overestimation
- **Double DQN** → `a* = argmax_a Q_online(s', a)` then `target = r + γ · Q_target(s', a*)` ← decoupled and accurate

The online network selects the action. The target network evaluates it. This single change leads to significantly more stable value estimates, especially in environments with large action spaces like Mario.

---

## 📊 Training Configuration

| Parameter | Value |
|---|---|
| Total Episodes | 600 |
| Max Steps per Episode | 500 |
| Batch Size | 32 |
| Learning Rate | 1e-4 (Adam) |
| Discount Factor γ | 0.99 |
| Replay Buffer Size | 30,000 |
| Replay Warm-up | 1,000 transitions |
| Target Network Sync | Every 1,000 steps |
| Framework | PyTorch |
| Environment | gym-super-mario-bros |
| Hardware | Google Colab T4 GPU |

---

## 🚀 How to Run

**Option 1 — Google Colab (Recommended)**

Open the notebook directly: [Click Here](https://colab.research.google.com/drive/1ncKQQKUFx0KtZuFt3xcW9PkEu8JlVjmf?usp=sharing)

Set runtime to GPU via **Runtime → Change runtime type → T4 GPU**, then click **Runtime → Run All**.

**Option 2 — Local Setup**

```bash
git clone https://github.com/mudassar2224/-Supe--Marios-Bros-using-Double-Deep-Q-Network.git
cd -Supe--Marios-Bros-using-Double-Deep-Q-Network
pip install gym==0.26.2 gym-super-mario-bros nes-py
pip install torch torchvision
pip install opencv-python imageio matplotlib tqdm
```

Then open and run the notebook cell by cell.

---

## 📁 Project Structure

📦 Super-Marios-Bros-DDQN

┣ 📓 Supe_Marios_Bros_using_Double_Deep_Q_Network.ipynb   ← Main training notebook

┣ 📄 README.md                                             ← Project documentation

┗ 📦 ddqn_mario.pth                                       ← Saved model weights


---

## 📚 References

- Mnih et al. (2015) — [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236), DeepMind / Nature
- van Hasselt et al. (2016) — [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), AAAI
- Kautenja — [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros), OpenAI Gym environment

---

## 👤 Author

**Muhammad Mudassar**
B.S. Artificial Intelligence — University of Management and Technology (UMT)
Course: AI473 — Artificial Neural Networks & Deep Learning, Spring 2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/muhmmad-mudassar-656384342/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/mudassar2224)
