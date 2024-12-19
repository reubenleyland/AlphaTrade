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
test_steps = 3000  # Adjusted for your test case; make sure this isn't too high

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
        "WINDOW_INDEX": 200,
        "ACTION_TYPE": "pure",
        "REWARD_LAMBDA": 0.1,
        "EP_TYPE": "fixed_time",
        "EPISODE_TIME": 60 * 50,  # 60 seconds
    }

    # Set up random keys for JAX
    rng = jax.random.PRNGKey(40)
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
        reward_lambda=0.0001,
        episode_time=config["EPISODE_TIME"],  # in seconds
    )

    # Initialize the environment state
    start = time.time()
    obs, state = env.reset(key_reset, env_params)
    print(f"Starting index in data: {state.start_index}")
    print("Time for reset: \n", time.time() - start)
    print("Inventory after reset: \n", state.inventory)
    print(f"Number of available windows: {env.n_windows}")

    # ============================
    # Initialize data storage
    # ============================
    reward_file = 'gymnax_exchange/test_scripts/test_outputs/data.csv'  # Relative path
    
    # Ensure the directory exists, if not, create it
    os.makedirs(os.path.dirname(reward_file), exist_ok=True)
    
    rewards = np.zeros((test_steps, 1), dtype=int)
    inventory = np.zeros((test_steps, 1), dtype=int)
    total_revenue = np.zeros((test_steps, 1), dtype=int)
    buyQuant = np.zeros((test_steps, 1), dtype=int)
    sellQuant = np.zeros((test_steps, 1), dtype=int)
    bid_price = np.zeros((test_steps, 1), dtype=int)
    ask_price = np.zeros((test_steps, 1), dtype=int)
    state_best_ask = np.zeros((test_steps, 1), dtype=int)
    state_best_bid = np.zeros((test_steps, 1), dtype=int)
    averageMidprice = np.zeros((test_steps, 1), dtype=int)
    average_best_bid =np.zeros((test_steps, 1), dtype=int)
    average_best_ask =np.zeros((test_steps, 1), dtype=int)
   

   # book_vol_av_bid= np.zeros((test_steps, 1), dtype=int)
   # book_vol_av_ask = np.zeros((test_steps, 1), dtype=int)

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
        test_action = jnp.array([1, 1])
        
        start = time.time()
        obs, state, reward, done, info = env.step(key_step, state, test_action, env_params)
        
        # Store data
        rewards[i] = reward
        inventory[i] = state.inventory
        total_revenue[i] = state.total_revenue
        buyQuant[i] = info["buyQuant"]
        sellQuant[i] = info["sellQuant"]
        bid_price[i] = info["action_prices_0"]  # Store best ask
        ask_price[i] = info["action_prices_1"]  # Store best bid
        averageMidprice[i] = info["averageMidprice"]  # Store mid price
        average_best_bid[i]=info["average_best_bid"]
        average_best_ask[i]=info["average_best_ask"]
        
    #    book_vol_av_bid[i]=info["book_vol_av_bid"]
     #   book_vol_av_ask[i]=info["book_vol_av_ask"]
       # state_best_ask[i] = state.best_asks[-1,0]
      #  state_best_bid[i] = state.best_bids[-1,0]
        # Increment valid steps
        valid_steps += 1
        
        if done:
            print("===" * 20)
            break

    # ============================
    # Clip the arrays to remove trailing zeros
    # ============================
    rewards = rewards[:valid_steps-1]
    inventory = inventory[:valid_steps-1]
    total_revenue = total_revenue[:valid_steps-1] 
    buyQuant = buyQuant[:valid_steps-1]
    sellQuant = sellQuant[:valid_steps-1]
    bid_price = bid_price[:valid_steps-1]
    ask_price = ask_price[:valid_steps-1]
    averageMidprice = averageMidprice[:valid_steps-1]
    average_best_bid =average_best_bid[:valid_steps-1]
    average_best_ask =average_best_ask[:valid_steps-1]
    #state_best_bid = state_best_bid[:valid_steps-1]
       # state_best_ask = state_best_ask[:valid_steps-1]
   # book_vol_av_bid= book_vol_av_bid[:valid_steps-1]
   # book_vol_av_ask= book_vol_av_ask[:valid_steps-1]

    # ============================
    # Save all data to CSV
    # ============================
    # Combine all data into a single 2D array (each column is one metric)
    data = np.hstack([rewards, inventory, total_revenue, buyQuant, sellQuant, bid_price, ask_price, averageMidprice])
    
    # Add column headers
    column_names = ['Reward', 'Inventory', 'Total Revenue', 'Buy Quantity', 'Sell Quantity', 'Bid Price', 'Ask Price', 'averageMidprice']
    
    # Save data using pandas to handle CSV easily
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(reward_file, index=False)
    
    print(f"Data saved to {reward_file}")

    # ============================
    # Plotting all metrics on one page
    # ============================
    # Create a figure with subplots (3 rows and 3 columns to fit the new data)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Adjust the grid as needed

    # Plot each metric on a separate subplot
    axes[0, 0].plot(range(valid_steps-1), rewards, label="Reward", color='blue')
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].set_title("Rewards Over Steps")
    
    axes[0, 1].plot(range(valid_steps-1), inventory, label="Inventory", color='green')
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Inventory")
    axes[0, 1].set_title("Inventory Over Steps")
    
    axes[0, 2].plot(range(valid_steps-1), total_revenue, label="Total Revenue", color='orange')
    axes[0, 2].set_xlabel("Steps")
    axes[0, 2].set_ylabel("Total Revenue")
    axes[0, 2].set_title("Total Revenue Over Steps")
    
    axes[1, 0].plot(range(valid_steps-1), buyQuant, label="Buy Quantity", color='red')
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("Buy Quantity")
    axes[1, 0].set_title("Buy Quantity Over Steps")
    
    axes[1, 1].plot(range(valid_steps-1), sellQuant, label="Sell Quantity", color='purple')
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Sell Quantity")
    axes[1, 1].set_title("Sell Quantity Over Steps")
    
    # Combined plot for Bid Price, Ask Price, and Average Mid Price
    axes[1, 2].plot(range(valid_steps-1), bid_price, label="Bid Price", color='pink')
    axes[1, 2].plot(range(valid_steps-1), ask_price, label="Ask Price", color='cyan')
    axes[1, 2].plot(range(valid_steps-1), averageMidprice, label="Average Mid Price", color='magenta')
    axes[1, 2].plot(range(valid_steps-1), average_best_bid, label="average_best_bid", color='red')
    axes[1, 2].plot(range(valid_steps-1), average_best_ask, label="average_best_ask", color='blue')
    axes[1, 2].set_xlabel("Steps")
    axes[1, 2].set_ylabel("Price")
    axes[1, 2].set_title("Bid, Ask & Mid Price Over Steps")
    axes[1, 2].legend()

    # Turn off the empty subplot (3, 3) position
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the combined plots as a single image
    combined_plot_file = 'gymnax_exchange/test_scripts/test_outputs/reward_complex.png'
    plt.savefig(combined_plot_file)
    plt.close()

    print(f"Combined plots saved to {combined_plot_file}")
