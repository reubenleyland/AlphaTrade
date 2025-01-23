import os
import sys
sys.path.append(os.path.abspath('/home/duser/AlphaTrade'))
sys.path.append('.')
import time
import dataclasses
import jax
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd  
import chex
from gymnax_exchange.jaxen.exec_env import ExecutionEnv
import faulthandler

faulthandler.enable()

# ============================
# Configuration
# ============================
test_steps = 15000  # Adjusted for your test case; make sure this isn't too high

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:", ATFolder)
    except:
        ATFolder = "./training_oneDay"
    
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "sell", # "random", # "buy",
        "MAX_TASK_SIZE": 500, # 500,
        "WINDOW_INDEX": 1,
        "ACTION_TYPE": "pure", # "pure",
        "REWARD_LAMBDA": 1.0,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60 * 60, # 60 seconds
    }
        
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # env=ExecutionEnv(ATFolder,"sell",1)
    env = ExecutionEnv(
        alphatradePath=config["ATFOLDER"],
        task=config["TASKSIDE"],
        window_index=config["WINDOW_INDEX"],
        action_type=config["ACTION_TYPE"],
        episode_time=config["EPISODE_TIME"],
        max_task_size=config["MAX_TASK_SIZE"],
        ep_type=config["EP_TYPE"],
    )
    # env_params=env.default_params
    env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=1,
        task_size=config["MAX_TASK_SIZE"],
        episode_time=config["EPISODE_TIME"],  # in seconds
    )
    

    # Initialize the environment state
    start = time.time()
    obs, state = env.reset(key_reset, env_params)
    print(f"Starting index in data: {state.start_index}")
    print("Time for reset: \n", time.time() - start)

    print(f"Number of available windows: {env.n_windows}")

    # ============================
    # Initialize data storage
    # ============================
    reward_file = 'gymnax_exchange/test_scripts/test_outputs/data_exec.csv'  # Relative path
    
    # Ensure the directory exists, if not, create it
    os.makedirs(os.path.dirname(reward_file), exist_ok=True)
    
    rewards = np.zeros((test_steps, 1), dtype=int)
    quant_executed = np.zeros((test_steps, 1), dtype=int)
    bid_price = np.zeros((test_steps, 1), dtype=int)
    ask_price = np.zeros((test_steps, 1), dtype=int)
    vwap=np.zeros((test_steps, 1), dtype=int)
 

    # ============================
    # Track the number of valid steps
    # ============================
    valid_steps = 0

    # ============================
    # Run the test loop
    # ============================
    for i in range(test_steps):
        # ==================== ACTION ====================
        key_policy, _ = jax.random.split(key_policy, 2)
        key_step, _ = jax.random.split(key_step, 2)
        test_action = jnp.array([0,1])
        
        start = time.time()
        obs, state, reward, done, info = env.step(key_step, state, test_action, env_params)
        
        # Store data
        rewards[i] = reward
        quant_executed[i] = state.quant_executed
        ask_price[i] = state.best_asks[-1,0]  # Store best ask
        bid_price[i] = state.best_bids[-1,0]
        vwap[i]=state.vwap_rm
       
    
        # Increment valid steps
        valid_steps += 1
        
        if done:
            print("===" * 20)
            break

   # ============================
# Clip the arrays to remove trailing zeros
# ============================
plot_until_step = valid_steps 

# Ensure that the slicing is correct
start_step = 0
end_step = plot_until_step

# Create the x-axis for plotting
x_axis = np.arange(start_step, end_step)

# Slice the relevant arrays
rewards = rewards[start_step:end_step]
quant_executed = quant_executed[start_step:end_step] 
bid_price = bid_price[start_step:end_step]
ask_price = ask_price[start_step:end_step]
vwap=vwap[start_step:end_step]

# ============================
# Save all data to CSV
# ============================
# Combine all data into a single 2D array (each column is one metric)
data = np.hstack([rewards,  bid_price, ask_price,vwap,quant_executed])

# Add column headers
column_names = ['rewards', 'bid_price', 'ask_price', 'vwap','quant_executed']

# Save data using pandas to handle CSV easily
df = pd.DataFrame(data, columns=column_names)
df.to_csv(reward_file, index=False)

print(f"Data saved to {reward_file}")
print(f"Last valid step {valid_steps}")

# ============================
# Plotting all metrics
# ============================
fig, axes = plt.subplots(3, 3, figsize=(18, 12))  # Adjust the grid as needed

# Plot bid price and ask price on the same graph
axes[0, 0].plot(x_axis, bid_price, label="Bid Price", color='blue')
axes[0, 0].plot(x_axis, ask_price, label="Ask Price", color='green')
axes[0, 0].set_xlabel("Steps")
axes[0, 0].set_ylabel("Price")
axes[0, 0].set_title("Bid & Ask Price Over Steps")
axes[0, 0].legend()

# Plot rewards over steps
axes[0, 1].plot(x_axis, rewards, label="Reward", color='blue')
axes[0, 1].set_xlabel("Steps")
axes[0, 1].set_ylabel("Reward")
axes[0, 1].set_title("Rewards Over Steps")

# Plot quantity executed over steps
axes[0, 2].plot(x_axis, quant_executed, label="Quantity Executed", color='purple')
axes[0, 2].set_xlabel("Steps")
axes[0, 2].set_ylabel("Quantity Executed")
axes[0, 2].set_title("Quantity Executed Over Steps")

axes[1, 0].plot(x_axis, vwap, label="VWAP", color='purple')
axes[1, 0].set_xlabel("Steps")
axes[1, 0].set_ylabel("VWAP")
axes[1, 0].set_title("VWAP")
    
# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the combined plots as a single image
combined_plot_file = 'gymnax_exchange/test_scripts/test_outputs/exec_test.png'
plt.savefig(combined_plot_file)
plt.close()
print(f"Combined plots saved to {combined_plot_file}")
