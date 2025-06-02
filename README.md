# Reinforcement Learning for 3D Eversion Robot

This project trains a 3D eversion robot using Stable Baselines3 (PPO). The environment simulates a soft robot navigating to a target while avoiding obstacles.

## 🛠 Setup

1. **Create a virtual environment** (Python 3.7+ recommended):
   ```bash
   python -m venv venv
   ```
   
2. **Activate the environment**:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Training the Model

Run the training script with PPO:
```bash
python training_eversion_robot_3d.py
```
*(Modify `training_eversion_robot_3d.py` to adjust hyperparameters like `total_timesteps` or policy network architecture.)*

## 🧪 Testing the Model

After training, test the saved model:
```bash
python testing_rl_eversion_robot_3d.py
```
*(Ensure `testing_rl_eversion_robot_3d.py` loads the correct model path.)*

## 📁 Project Structure
```
.
├── venv/                     # Virtual environment (ignored in Git)
├── eversion_robot_3d.py      # Custom Gymnasium environment
├── training_eversion_robot_3d.py  # Training script
├── testing_rl_eversion_robot_3d.py # Testing script
└── README.md                 # This file
```
