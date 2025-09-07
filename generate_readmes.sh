#!/usr/bin/env bash
set -e

# Top-level README.md
cat > README.md << 'EOF'
# Quad Move

Code Authors: **Advait Desai** and **Gargi Gupta**  
Project Mentors: **Ansh Semwal** and **Prajwal Awhad**  
This project is part of SRA VJTI's Eklavya 2025 Program.

Train a PPO-based gait policy in MuJoCo and deploy it on a low-cost, tortoise-style quadruped.

## 📁 Subprojects

1. Brax Training Viewer       → `1_brax_training_viewer/README.md`  
2. Mujoco Menagerie          → `2_mujoco_menagerie/README.md`  
3. Monte Carlo               → `3_monte_carlo/README.md`  
4. Q-Learning                → `4_Q_learning/README.md`  
5. Number Classifier         → `5_number_classifier/README.md`  
6. DQN                       → `6_DQN/README.md`  
7. DDQN                      → `7_DDQN/README.md`  
8. PPO – Bipedal             → `8_PPO_bipedal/README.md`  
9. PPO – Go2                 → `9_PPO_go2/README.md`

Run \`./generate_readmes.sh\` to regenerate all subproject READMEs automatically!
EOF

# Array of subproject directories and their README contents
declare -A projects=(
  ["1_brax_training_viewer"]="![Multi-ant Viewer](pre_gifs/multi_ants.gif)

\`\`\`\`bash
mjpython 1_ant-viewer.py
mjpython 2_multi-ant-viewer.py
\`\`\`\`"
  ["2_mujoco_menagerie"]="![Mujoco Flybody](pre_gifs/flybody.gif)

\`\`\`\`bash
python3 main.py
\`\`\`\`"
  ["3_monte_carlo"]="![Frozen Lake](pre_gifs/frozen_lake.gif) ![Blackjack](pre_gifs/blackjack.gif)

\`\`\`\`bash
python3 frozen_lake.py
\`\`\`\`"
  ["4_Q_learning"]="![Cart Pole](pre_gifs/cart_pole.gif) ![Mountain Car](pre_gifs/mountain_car.gif)

\`\`\`\`bash
python3 blackjack.py
\`\`\`\`"
  ["5_number_classifier"]="**Number Classifier Demo**

\`\`\`\`bash
# Add your number classifier demo commands here
\`\`\`\`"
  ["6_DQN"]="![Lunar Lander DQN](pre_gifs/lunar_lander.gif)

\`\`\`\`bash
python3 lunar_lander_visualise.py
\`\`\`\`"
  ["7_DDQN"]="![Lunar Lander DDQN](pre_gifs/lunar_lander.gif)

\`\`\`\`bash
python3 lunar_lander_gpu_ddqn.py
\`\`\`\`"
  ["8_PPO_bipedal"]="![Bipedal Walker](pre_gifs/bipedal_walker.gif)

\`\`\`\`bash
python3 biped.py   # to train
python3 test_biped.py   # to visualise
\`\`\`\`"
  ["9_PPO_go2"]="**Walk Demo**

\`\`\`\`bash
python3 train_go2.py   # to train
python3 test_go2.py    # to visualise
\`\`\`\`"
)

# Loop through each project and write its README.md
for dir in "${!projects[@]}"; do
  mkdir -p "$dir"
  cat > "$dir/README.md" << EOF
# $(echo "$dir" | sed 's/_/ /g')

${projects[$dir]}
EOF
done

echo "All README.md files generated successfully!"