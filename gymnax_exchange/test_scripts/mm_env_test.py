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
from gymnax_exchange.jaxen.mm_env import MarketMakingEnv
import faulthandler
import pandas as pd  # Adding pandas for easier CSV writing

faulthandler.enable()

# ============================
# Configuration
# ============================
test_steps = 1000

if __name__ == "__main__":
    try:
        ATFolder = sys.argv[1]
        print("AlphaTrade folder:", ATFolder)
    except:
        ATFolder = "./training_oneDay"
    
    config = {
        "ATFOLDER": ATFolder,
        "TASKSIDE": "buy",
        "MAX_TASK_SIZE": 100,
        "WINDOW_INDEX": 1,
        "ACTION_TYPE": "pure",
        "REWARD_LAMBDA": 1.0,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60 * 50,  # 60 seconds
    }

    # Set up random keys for JAX
    rng = jax.random.PRNGKey(0)
    rng, key_reset, key_policy, key_step = jax.random.split(rng, 4)

    # Initialize the environment
    env = MarketMakingEnv(
        alphatradePath=config["ATFOLDER"],
        task=config["TASKSIDE"],
        window_index=config["WINDOW_INDEX"],
        action_type=config["ACTION_TYPE"],
        episode_time=config["EPISODE_TIME"],
        max_task_size=config["MAX_TASK_SIZE"],
        ep_type=config["EP_TYPE"],
    )
    # Define environment parameters
    env_params = dataclasses.replace(
        env.default_params,
        reward_lambda=0.1,
        episode_time=config["EPISODE_TIME"],  # in seconds
    )

    # Initialize the environment state
    start = time.time()
    obs, state = env.reset(key_reset, env_params)
    print("Time for reset: \n", time.time() - start)
    print("Inventory after reset: \n", state.inventory)

    # ============================
    # Initialize data storage
    # ============================
    reward_file = 'gymnax_exchange/test_scripts/test_outputs/data.csv'
    
    rewards = np.zeros((test_steps, 1), dtype=int)
    inventory = np.zeros((test_steps, 1), dtype=int)
    total_revenue = np.zeros((test_steps, 1), dtype=int)
    buyQuant = np.zeros((test_steps, 1), dtype=int)
    sellQuant = np.zeros((test_steps, 1), dtype=int)

    # ============================
    # Run the test loop
    # ============================
    for i in range(1, test_steps):
        # ==================== ACTION ====================
        
        key_policy, _ = jax.random.split(key_policy, 2)
        key_step, _ = jax.random.split(key_step, 2)
        test_action = jnp.array([1, 1])
        #print(f"Sampled {i}th actions are: ", test_action)
        
        start = time.time()
        obs, state, reward, done, info = env.step(key_step, state, test_action, env_params)
        
        # Store data
        rewards[i] = reward
        inventory[i] = state.inventory
        total_revenue[i] = state.total_revenue
        buyQuant[i] = info["buyQuant"]
        sellQuant[i] = info["sellQuant"]

        if done:
            print("===" * 20)
            break

    # ============================
    # Save all data to CSV
    # ============================
    # Combine all data into a single 2D array (each column is one metric)
    data = np.hstack([rewards, inventory, total_revenue, buyQuant, sellQuant])
    
    # Add column headers
    column_names = ['Reward', 'Inventory', 'Total Revenue', 'Buy Quantity', 'Sell Quantity']
    
    # Save data using pandas to handle CSV easily
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(reward_file, index=False)
    
    print(f"Data saved to {reward_file}")

  # ============================
    # Plotting all metrics on one page
    # ============================
    # Create a figure with subplots (2 rows and 3 columns for example)
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))  # Adjust the grid as needed

    # Plot each metric on a separate subplot
    axes[0, 0].plot(range(test_steps), rewards, label="Reward", color='blue')
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("Rewards Over Steps")
    
    axes[0, 1].plot(range(test_steps), inventory, label="Inventory", color='green')
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Inventory")
    axes[0, 1].set_title("Inventory Over Steps")
    
    axes[1, 0].plot(range(test_steps), total_revenue, label="Total Revenue", color='orange')
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("Total Revenue")
    axes[1, 0].set_title("Total Revenue Over Steps")
    
    axes[1, 1].plot(range(test_steps), buyQuant, label="Buy Quantity", color='red')
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Buy Quantity")
    axes[1, 1].set_title("Buy Quantity Over Steps")
    
    axes[2, 0].plot(range(test_steps), sellQuant, label="Sell Quantity", color='purple')
    axes[2, 0].set_xlabel("Steps")
    axes[2, 0].set_ylabel("Sell Quantity")
    axes[2, 0].set_title("Sell Quantity Over Steps")
    
    # Turn off the empty subplot (3, 2) position
    axes[2, 1].axis('off')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the combined plots as a single image
    combined_plot_file = 'gymnax_exchange/test_scripts/test_outputs/plots.png'
    plt.savefig(combined_plot_file)
    plt.close()

    print(f"Combined plots saved to {combined_plot_file}")
