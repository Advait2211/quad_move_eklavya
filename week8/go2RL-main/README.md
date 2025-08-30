# go2RL

### Setting up the environment

1. Create a workspace folder

2. Clone this repository inside the workspace folder 

3. Create a virtual environment

```
python3 -m venv <name of virtual environment>
```

4. Activate the virtual environment

```
source <name of virtual environment>/bin/activate
```

5. Install necessary libraries

```
pip install -r requirements.txt
```

6. Inside the workspace folder, clone the rsl_rl repository found here https://github.com/leggedrobotics/rsl_rl/tree/main

7. Install the python package

```
cd rsl_rl && git checkout v1.0.2 && pip install -e .
pip install tensorboard
```

8. Go back to the root of the workspace

```
cd ..
```

9. Clone the genesis repository found here https://github.com/Genesis-Embodied-AI/Genesis/tree/main


10. Navigate to this go2RL directory

```
cd go2RL/
```

11. Run the training script

```
python go2_train.py
```

12. Run the evaluation script

```
python go2_eval.py
```

