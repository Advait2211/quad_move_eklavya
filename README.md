# Quad Move

Code Authors: **Advait Desai** and **Gargi Gupta**  
Project Mentors: **Ansh Semwal** and **Prajwal Awhad**  
This project is part of **SRA VJTI's Eklavya 2025 Program**  

---

## 📖 About the Project
We aim to train a PPO-based gait policy in MuJoCo and deploy it on a low-cost, tortoise-style quadruped robot.  

The project flow:  
1. Learn Reinforcement Learning basics (Monte Carlo, Q-Learning)  
2. Learn Deep Learning (MNIST digit classifier)  
3. Combine Deep Learning with RL (DQN, DDQN, TRPO, PPO)  
4. Implement on environments (Brax, MuJoCo Menagerie, Bipedal, Go2)  

---

## 📂 File Structure

quad_move_eklavya
├── 1_brax_training_viewer
├── 2_mujoco_menagerie
│ └── flybody
├── 3_monte_carlo
├── 4_Q_learning
├── 5_number_classifier
├── 6_DQN
├── 7_DDQN
├── 8_PPO_bipedal
└── 9_PPO_go2_stable_baselines


---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Advait2211/quad_move_eklavya.git
cd quad_move_eklavya

2️⃣ Create a Virtual Environment

python3.10 -m venv venv

3️⃣ Activate the Virtual Environment

    macOS/Linux

source venv/bin/activate

Windows (PowerShell)

    .\venv\Scripts\activate

4️⃣ Install Dependencies

pip install -r requirements.txt