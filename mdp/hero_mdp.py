from functools import cache
from memo import memo
import jax
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

hidden_mu_vals = np.array([95.0, 105.0, 95.0]) # A, B, C
trader_names = ["A", "B", "C"]
num_turns = 3

# True market value of the asset (used for reward calculation)
TRUE_MU = 100.0

# Initial Market State
starting_bid = 90.0
starting_ask = 110.0
num_prices = int(starting_ask - starting_bid + 1)

# State Space: Bid_Index * Ask_Index * Turn_Index
S = np.arange(num_prices * num_prices * num_turns)

# Actions: 0: tighten bid (+1) & ask (-1), 1: trade buy, 2: trade sell
A = np.array([0, 1, 2])


@jax.jit
def decode(s):
    """Decodes state index into (bid_idx, ask_idx, turn_idx)"""
    bi = s % num_prices
    ai = (s // num_prices) % num_prices
    ti = s // (num_prices * num_prices)
    return bi, ai, ti

@jax.jit
def encode(bi, ai, ti):
    """Encodes (bid_idx, ask_idx, turn_idx) into state index"""
    return bi + (ai * num_prices) + (ti * num_prices * num_prices)

@jax.jit
def get_price_vals(bi, ai):
    """Converts 0-based indices back to float prices"""
    bid_val = bi + starting_bid
    ask_val = ai + starting_bid
    return bid_val, ask_val

@jax.jit
def get_utility(s_next, a_candidate, future_q_val):
    """
    Returns the value to maximize.
    - If Hero's turn: Returns 'future_q_val' (Standard MDP).
    - If Opponent's turn: Returns 0.0 if Greedy, -1e9 if Not Greedy.
    """
    ti_next = s_next // (num_prices * num_prices)
    is_hero_turn = (ti_next == 0)

    # Calculate what the greedy opponent would do
    greedy_a = get_greedy_action_jax(s_next)
    is_greedy_move = (a_candidate == greedy_a)

    return np.where(
        is_hero_turn,
        future_q_val,
        np.where(is_greedy_move, future_q_val, -1.0e9) # has to be greedy
    )

@jax.jit
def is_terminating(s, a):
    """Round ends if a Trade occurs"""
    return (a == 1) | (a == 2)

@jax.jit
def get_greedy_action_jax(s):
    """
    JAX implementation of your 'get_greedy_action' logic.
    Used to predict what Agents B and C will do.
    """
    bi, ai, ti = decode(s)
    bid_val, ask_val = get_price_vals(bi, ai)
    mu_i = hidden_mu_vals[ti] # Use the specific agent's hidden valuation

    can_buy = (mu_i - ask_val) > 0
    can_sell = (bid_val - mu_i) > 0

    return np.where(can_buy, 1,
           np.where(can_sell, 2, 0)
           )

@jax.jit
def Tr(s, a, s_):
    """
    Transition Function P(s' | s, a).
    """
    bi, ai, ti = decode(s)
    ti_next = (ti + 1) % num_turns
    valid_tighten = (bi + 1) < (ai - 1)
    bi_next = np.where((a == 0) & valid_tighten, bi + 1, bi)
    ai_next = np.where((a == 0) & valid_tighten, ai - 1, ai)
    expected_s_ = encode(bi_next, ai_next, ti_next)
    return 1.0 * (s_ == expected_s_)

@jax.jit
def R(s, a):
    """
    Reward logic for the PLANNER (Hero/Agent A).
    Includes both ACTIVE trading profit and PASSIVE counterparty profit.
    """
    bi, ai, ti = decode(s)
    bid_val, ask_val = get_price_vals(bi, ai)

    is_hero_turn = (ti == 0)
    is_hero_counterparty = (ti == 1)
    hero_mu = hidden_mu_vals[0]

    hero_active_r = np.where(a == 1, hero_mu - ask_val,
                    np.where(a == 2, bid_val - hero_mu,
                    0.0))

    hero_passive_r = np.where(a == 1, ask_val - hero_mu, # Opponent Bought at Ask
                     np.where(a == 2, hero_mu - bid_val, # Opponent Sold at Bid
                     0.0))

    return np.where(is_hero_turn, hero_active_r,
           np.where(is_hero_counterparty, hero_passive_r,
           0.0))

@cache
@memo
def Q[s: S, a: A](t):
    agent: knows(s, a)
    agent: given(s_ in S, wpp=Tr(s, a, s_))
    agent: chooses(
        a_ in A,
        to_maximize=(
            0.0 if t < 0 else
            0.0 if is_terminating(s, a) else
            get_utility(s_, a_, Q[s_, a_](t - 1))
        )
    )
    return E[
        R(s, a) + (
            0.0 if t < 0 else
            0.0 if is_terminating(s, a) else
            Q[agent.s_, agent.a_](t - 1)
        )
    ]

def visualize_optimal_policy(Q_vals, history=None, filename='optimal_policy_clean.png'):
    """
    Visualizes the policy with ILLEGAL states (Bid >= Ask) masked out (transparent).
    """
    Q_array = onp.array(Q_vals)

    # 1. Create the Policy Grid
    # Initialize with a dummy value (e.g., 0)
    policy_grid = onp.zeros((num_prices, num_prices))

    # Fill with optimal actions
    for s in range(len(S)):
        bi, ai, ti = decode(s)
        bi, ai, ti = int(bi), int(ai), int(ti)

        if ti == 0: # Planner Only
            q_s = Q_array[s, :]
            optimal_action = int(onp.argmax(q_s))
            if ai < num_prices and bi < num_prices:
                policy_grid[ai, bi] = optimal_action

    # 2. Create a Mask for Illegal States (Ask <= Bid)
    # We create a boolean grid where True = "Hide this cell"
    mask = onp.zeros_like(policy_grid, dtype=bool)
    for bi in range(num_prices):
        for ai in range(num_prices):
            if ai <= bi:
                mask[ai, bi] = True # Mask out illegal states

    # Apply the mask
    masked_policy_grid = onp.ma.masked_array(policy_grid, mask)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(12, 10))

    # Updated Colormap: Only needs 3 colors now (Tighten, Buy, Sell)
    # 0=Tighten (Blue), 1=Buy (Green), 2=Sell (Red)
    from matplotlib.colors import ListedColormap, BoundaryNorm
    policy_colors = ['#3498db', '#2ecc71', '#e74c3c']
    cmap = ListedColormap(policy_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5] # Boundaries between 0, 1, 2
    norm = BoundaryNorm(bounds, cmap.N)

    # Extent maps array indices to price values
    extent = [starting_bid - 0.5, starting_ask + 0.5, starting_bid - 0.5, starting_ask + 0.5]

    # Render with 'set_bad' color (transparency)
    cmap.set_bad(color='white', alpha=0)

    im = ax.imshow(masked_policy_grid, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest', origin='lower', extent=extent)

    if history:
        path_bids = [step['bid'] for step in history]
        path_asks = [step['ask'] for step in history]

        ax.plot(path_bids, path_asks, color='black', linewidth=3, linestyle='-', alpha=0.3, label='Path')
        ax.plot(path_bids, path_asks, color='white', linewidth=1.5, linestyle='--')

        # Start icon: Star marker with gold color
        ax.scatter(path_bids[0], path_asks[0], color='gold', s=400, marker='*',
                  edgecolors='black', linewidths=2.5, zorder=10, label='Start')
        # Add text annotation for clarity
        ax.annotate('START', xy=(path_bids[0], path_asks[0]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='gold', alpha=0.9, edgecolor='black', linewidth=1.5),
                   zorder=11)

        # Trade/End icon: Different markers based on final action
        final_act = history[-1]['action_name']
        final_bid, final_ask = path_bids[-1], path_asks[-1]

        # Use different icons based on final action
        if final_act in ['BUY', 'SELL']:
            # Trade completed: Use filled circle with checkmark or dollar sign
            ax.scatter(final_bid, final_ask, color='green', s=500, marker='o',
                      edgecolors='darkgreen', linewidths=3, zorder=10, label=f'Trade ({final_act})')
            # Add dollar sign or checkmark as text overlay
            trade_symbol = '$' if final_act == 'BUY' else '✓'
            ax.text(final_bid, final_ask, trade_symbol, fontsize=16, fontweight='bold',
                   color='white', ha='center', va='center', zorder=11)
            ax.annotate(f'TRADE\n({final_act})', xy=(final_bid, final_ask),
                       xytext=(8, -20), textcoords='offset points',
                       fontsize=9, fontweight='bold', color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.9, edgecolor='darkgreen', linewidth=1.5),
                       ha='left', zorder=11)
        else:
            # No trade: Use X marker
            ax.scatter(final_bid, final_ask, color='red', s=400, marker='X',
                      edgecolors='darkred', linewidths=3, zorder=10, label=f'End ({final_act})')
            ax.annotate(f'END\n({final_act})', xy=(final_bid, final_ask),
                       xytext=(8, -20), textcoords='offset points',
                       fontsize=9, fontweight='bold', color='darkred',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.9, edgecolor='darkred', linewidth=1.5),
                       ha='left', zorder=11)

    # Formatting
    hero_mu = float(hidden_mu_vals[0])
    ax.axvline(hero_mu, color='purple', linestyle=':', alpha=0.5, label=f'Hero Val ({hero_mu})')
    ax.axhline(hero_mu, color='purple', linestyle=':', alpha=0.5)

    ax.set_xlabel('Bid Price')
    ax.set_ylabel('Ask Price')
    ax.set_title('Optimal Policy')

    # Legend
    patches = [
        mpatches.Patch(color='#3498db', label='Tighten'),
        mpatches.Patch(color='#2ecc71', label='BUY'),
        mpatches.Patch(color='#e74c3c', label='SELL')
    ]
    ax.legend(handles=patches, loc='upper right')

    plt.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

def run_simulation(Q_vals, visualize=True, t_horizon=10):
    curr_bid = starting_bid
    curr_ask = starting_ask
    curr_turn = 0 # Start with Agent A
    no_progress_count = 0

    true_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
    hidden_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}

    # Track history for visualization
    history = []

    print(f"\n--- SIMULATION START: Bid {curr_bid} | Ask {curr_ask} | True Value {TRUE_MU} ---")

    for step in range(15):
        trader = trader_names[curr_turn]

        prev_trader_idx = (curr_turn - 1) % 3
        prev_trader = trader_names[prev_trader_idx]

        b_idx = int(curr_bid - starting_bid)
        a_idx = int(curr_ask - starting_bid)
        s_idx = encode(b_idx, a_idx, curr_turn)

        if curr_turn == 0:
            # --- AGENT A (HERO) uses Planned Policy ---
            q_s = Q_vals[s_idx, :]

            # # Filter Invalid Tightens (cannot cross spread)
            # # If bid+1 >= ask-1, we cannot tighten further without locking
            # if (curr_bid + 1) > (curr_ask - 1):
            #     q_s = q_s.at[0].set(-1e9) # Block Tighten

            action = int(np.argmax(q_s))
            source = "Planned (MDP)"
        else:
            # --- AGENTS B/C use Greedy Policy ---
            action = int(get_greedy_action_jax(s_idx))
            source = f"Greedy (Mu={hidden_mu_vals[curr_turn]})"

        # Log the attempt
        act_str = ["Tighten", "BUY", "SELL"][action]
        print(f"Step {step} | {trader} [{source}] decides to {act_str}")

        # Record history
        history.append({
            'step': step,
            'trader': trader,
            'bid': curr_bid,
            'ask': curr_ask,
            'action': action,
            'action_name': act_str
        })

        # 3. EXECUTE & DISTRIBUTE REWARDS
        if action == 1: # BUY
            # Trader Buys @ Ask from Counterparty
            r_actor = TRUE_MU - curr_ask
            r_counter = curr_ask - TRUE_MU
            r_actor_hidden = float(hidden_mu_vals[curr_turn]) - curr_ask
            r_counter_hidden = curr_ask - float(hidden_mu_vals[prev_trader_idx])

            true_rewards[trader] += r_actor
            true_rewards[prev_trader] += r_counter
            hidden_rewards[trader] += r_actor_hidden
            hidden_rewards[prev_trader] += r_counter_hidden

            print(f" >>> EXECUTION: {trader} BUYS from {prev_trader} @ {curr_ask}")
            print(f"     {trader} Profit: {r_actor} (TrueMu {TRUE_MU} - Ask {curr_ask})")
            print(f"     {prev_trader} Profit: {r_counter} (Ask {curr_ask} - TrueMu {TRUE_MU})")
            break # Round Ends

        elif action == 2: # SELL
            # Trader Sells @ Bid to Counterparty
            r_actor = curr_bid - TRUE_MU
            r_counter = TRUE_MU - curr_bid
            r_actor_hidden = curr_bid - float(hidden_mu_vals[curr_turn])
            r_counter_hidden = float(hidden_mu_vals[prev_trader_idx]) - curr_bid

            true_rewards[trader] += r_actor
            true_rewards[prev_trader] += r_counter
            hidden_rewards[trader] += r_actor_hidden
            hidden_rewards[prev_trader] += r_counter_hidden

            print(f" >>> EXECUTION: {trader} SELLS to {prev_trader} @ {curr_bid}")
            print(f"     {trader} Profit: {r_actor} (Bid {curr_bid} - TrueMu {TRUE_MU})")
            print(f"     {prev_trader} Profit: {r_counter} (TrueMu {TRUE_MU} - Bid {curr_bid})")
            break # Round Ends

        elif action == 0: # TIGHTEN
            if (curr_bid + 1) < (curr_ask - 1):
                curr_bid += 1
                curr_ask -= 1
                print(f"     Update: Spread tightens to {curr_bid} ... {curr_ask}")
            else:
                no_progress_count += 1

                if no_progress_count >= 3:
                    print(f"\n >>> TERMINATION: No progress made for {no_progress_count} consecutive steps.")
                    print(f"     Spread locked at {curr_bid} ... {curr_ask}. No trade executed.")
                    break

        curr_turn = (curr_turn + 1) % 3

    print("\n" + "="*40)
    print("FINAL SCORES (Based on True Rewards)")
    print(true_rewards)
    print("FINAL SCORES (Based on Hidden Rewards)")
    print(hidden_rewards)
    print("="*40)

    # Generate visualizations
    if visualize:
        visualize_optimal_policy(Q_vals)  # Focused policy visualization

    return history, true_rewards, hidden_rewards

if __name__ == "__main__":
    print("Compiling Policy...")
    t_horizon = 10
    Q_vals = Q(t_horizon).block_until_ready()
    print("Policy compiled. Running simulation...")

    # 1. Run Sim
    history, true_rewards, hidden_rewards = run_simulation(Q_vals, visualize=False, t_horizon=t_horizon)

    # 2. Visualize with History Overlay
    visualize_optimal_policy(Q_vals, history=history)
