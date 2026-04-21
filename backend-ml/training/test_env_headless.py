import sys
import os

# 1. Fix Pathing: Ensure Python can see the backend-ml root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 2. Import the actual class from your env file
try:
    from shadowops_env import UniversalShadowEnv
except ImportError:
    # Fallback for different folder structures
    from environment.shadowops_env import UniversalShadowEnv

def run_headless_test():
    print("=== SHADOWOPS HEADLESS ENVIRONMENT TEST ===\n")
    
    # Initialize the engine
    env = UniversalShadowEnv(mode="live", seed=99)
    
    # Unpack the initial reset (String, Array)
    _, obs = env.reset()
    print(f"Initial State | OBS_DIM: {len(obs)} | Risk Features Loaded\n")

    # 3. Define the test sequence
    # 3=QUARANTINE, 2=FORK, 0=ALLOW
    test_actions = [
        (3, "Trigger QUARANTINE (Step 1)"),
        (3, "Hold QUARANTINE (Step 2)"), 
        (3, "Hold QUARANTINE (Step 3 - Forces Resolution)"),
        (2, "Resolve to FORK (Attacker Trapped)")
    ]

    for step_num, (action, description) in enumerate(test_actions, 1):
        print(f"--- Step {step_num}: {description} ---")
        
        # 4. Correct Unpacking based on your env's return order:
        # (obs_text, obs_vec, reward, done, info)
        _, next_obs, reward, done, info = env.step(action)
        
        # Verify 18-dim features (last two: active flag and steps)
        q_active = next_obs[-2] 
        q_steps = next_obs[-1]
        
        print(f"Action Executed: {action}")
        print(f"Reward Emitted : {reward}")
        print(f"Quarantine Status: Active={q_active}, Steps={q_steps}")
        print(f"Episode Done   : {done}\n")

if __name__ == "__main__":
    run_headless_test()