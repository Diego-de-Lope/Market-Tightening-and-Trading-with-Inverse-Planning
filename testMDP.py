"""
Memo probabilistic program for Trade or Tighten MDP.
"""
from functools import cache
from memo import memo
import jax
import jax.numpy as np
import numpy as onp  # Original numpy for non-JAX operations
import matplotlib.pyplot as plt
from collections import defaultdict

# example
bids = np.arange(98, 103, 1)
asks = np.arange(98, 103, 1)
# mu_values = [float(m) for m in np.arange(90, 111, 1)]
mu_values = [float(m) for m in np.arange(98, 103, 1)]

NB = len(bids)
NA = len(asks)
NT = 3

S = np.arange(NB * NA * NT)
A = np.array([0, 1, 2, 3])
MU = np.array(mu_values)

@jax.jit
def decode(s):
    bi = s % NB
    ai = (s // NB) % NA
    ti = s // (NB * NA)
    return bi, ai, ti

@jax.jit
def encode(bi, ai, ti):
    return bi + NB * ai + NB * NA * ti

@jax.jit
def is_trade(a):
    return (a == 2) | (a == 3)

@jax.jit
def reward(s, a, mu):
    bi, ai, _ = decode(s)
    bid = bids[bi]
    ask = asks[ai]
    return np.where(a == 2, mu - ask,
           np.where(a == 3, bid - mu, 0.0))

@jax.jit
def transition_deterministic(s, a):
    bi, ai, ti = decode(s)
    ti2 = (ti + 1) % NT

    # default
    bi2 = bi
    ai2 = ai

    # tighten bid
    bi2 = np.where((a == 0) & ((bi + 1) < ai), bi + 1, bi2)
    # tighten ask
    ai2 = np.where((a == 1) & (bi < (ai - 1)), ai - 1, ai2)

    # trade -> reset spread
    bi2 = np.where(is_trade(a), 0, bi2)
    ai2 = np.where(is_trade(a), NA - 1, ai2)

    # If you want to forbid bi >= ai states, clamp them (or make them sticky)
    # Here: make illegal states stay in place
    illegal = bi2 >= ai2
    bi2 = np.where(illegal, bi, bi2)
    ai2 = np.where(illegal, ai, ai2)
    ti2 = np.where(illegal, ti, ti2)

    return encode(bi2, ai2, ti2)

@jax.jit
def Tr(s, a, s_):
    return 1.0 * (transition_deterministic(s, a) == s_)

@jax.jit
def is_terminating(s, a):
    return is_trade(a)

@cache
@memo
def Q[s: S, a: A, mu: MU](t):
    agent: knows(s, a, mu)
    agent: given(s_ in S, wpp=Tr(s, a, s_))
    agent: chooses(
        a_ in A,
        to_maximize=(
            0.0 if (t < 0) else
            0.0 if is_terminating(s, a) else
            Q[s_, a_, mu](t - 1)
        )
    )
    return E[
        reward(s, a, mu)
        + (
            0.0 if (t < 0) else
            0.0 if is_terminating(s, a) else
            Q[agent.s_, agent.a_, mu](t - 1)
        )
    ]

# ============================================================================
# Policy and Simulation Functions
# ============================================================================

# def get_q_value(q_dist, s, a, mu):
#     """
#     Extract Q-value from Memo distribution.

#     Args:
#         q_dist: Distribution returned by Q(t)
#         s, a, mu: State, action, belief values (as integers/arrays)

#     Returns:
#         q_value: Q-value as float
#     """
#     try:
#         # Try to access the distribution directly
#         # Memo distributions may support direct indexing
#         if hasattr(q_dist, '__getitem__'):
#             try:
#                 # Try 3D indexing
#                 return float(q_dist[s, a, mu])
#             except:
#                 try:
#                     # Try as a flat array
#                     s_idx = int(s) if hasattr(s, '__int__') else s
#                     a_idx = int(a) if hasattr(a, '__int__') else a
#                     mu_idx = int(mu) if hasattr(mu, '__int__') else mu
#                     flat_idx = s_idx * len(A) * len(MU) + a_idx * len(MU) + mu_idx
#                     return float(q_dist[flat_idx])
#                 except:
#                     pass
#         # Fallback: return 0 if we can't extract value
#         return 0.0
#     except Exception as e:
#         # If all else fails, return 0
#         return 0.0

# def get_valid_actions(s):
#     """
#     Get valid actions for current state.

#     Args:
#         s: Encoded state (integer)

#     Returns:
#         valid_actions: List of valid action indices
#     """
#     bi, ai, ti = decode(s)
#     bi, ai, ti = int(bi), int(ai), int(ti)
#     valid = []

#     # Tighten bid (if possible)
#     if (bi + 1) < ai:
#         valid.append(0)

#     # Tighten ask (if possible)
#     if bi < (ai - 1):
#         valid.append(1)

#     # Trade buy (if ask exists - always possible if ai < NA)
#     if ai < NA:
#         valid.append(2)

#     # Trade sell (if bid exists - always possible if bi < NB)
#     if bi < NB:
#         valid.append(3)

#     return valid if valid else [0, 1]  # Default to tighten actions

# def policy_softmax(s, mu, t, beta=1.0, q_dist=None):
#     """
#     Select action using softmax over Q-values.

#     Args:
#         s: Current state (encoded integer)
#         mu: Agent's belief about asset value
#         t: Time horizon
#         beta: Temperature parameter (higher = more rational)
#         q_dist: Pre-computed Q distribution (optional)

#     Returns:
#         action: Selected action index
#     """
#     if q_dist is None:
#         q_dist = Q(t)

#     # Find closest mu value in MU array
#     mu_idx = int(np.argmin(np.abs(MU - mu)))
#     mu_val = float(MU[mu_idx])

#     # Get valid actions
#     valid_actions = get_valid_actions(s)

#     # Get Q-values for all valid actions
#     q_values = []
#     for a in valid_actions:
#         q_val = get_q_value(q_dist, s, a, mu_val)
#         q_values.append(q_val)

#     if not q_values:
#         # Fallback: return first valid action
#         return valid_actions[0] if valid_actions else 0

#     q_values = np.array(q_values)

#     # Softmax with temperature
#     exp_q = np.exp(beta * q_values)
#     probs = exp_q / np.sum(exp_q)

#     # Convert to numpy array for sampling
#     probs_np = onp.array(probs)
#     probs_np = probs_np / probs_np.sum()  # Normalize

#     # Sample action
#     action_idx = onp.random.choice(len(valid_actions), p=probs_np)
#     return valid_actions[action_idx]

# def simulate_game(num_rounds, agents_mu, initial_state=None, t_horizon=5, beta=1.0, seed=42):
#     """
#     Simulate a full game with multiple traders.

#     Args:
#         num_rounds: Number of trading rounds to simulate
#         agents_mu: List of 3 belief values [mu_A, mu_B, mu_C]
#         initial_state: Initial encoded state (None = default)
#         t_horizon: Time horizon for Q-function
#         beta: Rationality parameter for policy
#         seed: Random seed

#     Returns:
#         history: List of game events
#         rewards: Dictionary of cumulative rewards per trader
#     """
#     onp.random.seed(seed)

#     # Initialize state
#     if initial_state is None:
#         # Start with wide spread, trader 0's turn
#         s = encode(0, NA - 1, 0)  # min bid, max ask, trader 0
#     else:
#         s = initial_state

#     history = []
#     rewards = {0: 0.0, 1: 0.0, 2: 0.0}  # Cumulative rewards per trader

#     # Pre-compute Q distribution
#     print(f"Computing Q-function with t={t_horizon}...")
#     q_dist = Q(t_horizon)
#     print("Q-function computed.")

#     round_num = 0
#     step = 0

#     while round_num < num_rounds:
#         bi, ai, ti = decode(s)
#         bi, ai, ti = int(bi), int(ai), int(ti)
#         trader_name = ['A', 'B', 'C'][ti]
#         mu = agents_mu[ti]

#         # Get valid actions
#         valid_actions = get_valid_actions(s)
#         if not valid_actions:
#             print(f"Warning: No valid actions at state {s}, ending game")
#             break

#         # Select action using policy
#         action = policy_softmax(s, mu, t_horizon, beta, q_dist)

#         # If action is invalid, pick first valid one
#         if action not in valid_actions:
#             action = valid_actions[0]

#         # Execute action
#         s_next = transition_deterministic(s, action)
#         r = float(reward(s, action, mu))
#         rewards[ti] += r

#         # Record event
#         action_names = {0: 'tighten_bid', 1: 'tighten_ask', 2: 'trade_buy', 3: 'trade_sell'}
#         event = {
#             'step': step,
#             'round': round_num,
#             'trader': trader_name,
#             'trader_idx': ti,
#             'state': (bi, ai, ti),
#             'bid_price': float(bids[bi]),
#             'ask_price': float(asks[ai]),
#             'spread': float(asks[ai] - bids[bi]),
#             'action': action_names[action],
#             'action_idx': int(action),
#             'reward': r,
#             'mu': mu,
#             'cumulative_reward': rewards[ti]
#         }
#         history.append(event)

#         # Check if trade occurred
#         if is_trade(action):
#             round_num += 1
#             event['round_ended'] = True
#         else:
#             event['round_ended'] = False

#         # Update state
#         s = s_next
#         step += 1

#         # Safety: prevent infinite loops
#         if step > 1000:
#             print("Warning: Max steps reached, ending game")
#             break

#     return history, rewards

# def plot_game_history(history, save_path=None):
#     """
#     Visualize game history with multiple plots.

#     Args:
#         history: List of game events from simulate_game
#         save_path: Optional path to save figure
#     """
#     if not history:
#         print("No history to plot")
#         return

#     fig, axes = plt.subplots(3, 1, figsize=(12, 10))

#     steps = [e['step'] for e in history]
#     bid_prices = [e['bid_price'] for e in history]
#     ask_prices = [e['ask_price'] for e in history]
#     spreads = [e['spread'] for e in history]
#     traders = [e['trader'] for e in history]
#     actions = [e['action'] for e in history]

#     # Track cumulative rewards per trader
#     cumulative_rewards = {'A': [], 'B': [], 'C': []}
#     cum_reward_A = 0
#     cum_reward_B = 0
#     cum_reward_C = 0

#     for e in history:
#         if e['trader'] == 'A':
#             cum_reward_A += e['reward']
#         elif e['trader'] == 'B':
#             cum_reward_B += e['reward']
#         else:
#             cum_reward_C += e['reward']

#         cumulative_rewards['A'].append(cum_reward_A)
#         cumulative_rewards['B'].append(cum_reward_B)
#         cumulative_rewards['C'].append(cum_reward_C)

#     # Plot 1: Bid/Ask Spread
#     ax1 = axes[0]
#     ax1.plot(steps, bid_prices, 'b-', label='Bid', linewidth=2)
#     ax1.plot(steps, ask_prices, 'r-', label='Ask', linewidth=2)
#     ax1.fill_between(steps, bid_prices, ask_prices, alpha=0.3, color='gray', label='Spread')

#     # Mark trades
#     trade_steps = [i for i, e in enumerate(history) if e['action'].startswith('trade')]
#     for ts in trade_steps:
#         ax1.axvline(x=steps[ts], color='green', linestyle='--', alpha=0.5)
#         trade_price = ask_prices[ts] if history[ts]['action'] == 'trade_buy' else bid_prices[ts]
#         ax1.scatter(steps[ts], trade_price, color='green', s=100, marker='*', zorder=5)

#     ax1.set_xlabel('Step')
#     ax1.set_ylabel('Price')
#     ax1.set_title('Bid/Ask Spread Over Time')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)

#     # Plot 2: Actions by Trader
#     ax2 = axes[1]
#     trader_colors = {'A': 'blue', 'B': 'orange', 'C': 'green'}
#     trader_y_pos = {'A': 0, 'B': 1, 'C': 2}

#     for i, (step, trader, action) in enumerate(zip(steps, traders, actions)):
#         color = trader_colors[trader]
#         marker = 'o' if 'tighten' in action else 's'
#         size = 100 if 'trade' in action else 50
#         y_pos = trader_y_pos[trader]
#         ax2.scatter(step, y_pos, c=color, marker=marker, s=size, alpha=0.7)

#     ax2.set_xlabel('Step')
#     ax2.set_ylabel('Trader')
#     ax2.set_title('Actions by Trader (circle=tighten, square=trade)')
#     ax2.set_yticks([0, 1, 2])
#     ax2.set_yticklabels(['A', 'B', 'C'])
#     ax2.grid(True, alpha=0.3)

#     # Plot 3: Cumulative Rewards
#     ax3 = axes[2]
#     ax3.plot(steps, cumulative_rewards['A'], 'b-', label='Trader A', linewidth=2)
#     ax3.plot(steps, cumulative_rewards['B'], 'r-', label='Trader B', linewidth=2)
#     ax3.plot(steps, cumulative_rewards['C'], 'g-', label='Trader C', linewidth=2)

#     ax3.set_xlabel('Step')
#     ax3.set_ylabel('Cumulative Reward')
#     ax3.set_title('Cumulative Rewards by Trader')
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)

#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150)
#         print(f"Figure saved to {save_path}")
#     else:
#         plt.show()

# def print_game_summary(history, rewards):
#     """
#     Print a text summary of the game.

#     Args:
#         history: List of game events
#         rewards: Dictionary of cumulative rewards per trader
#     """
#     print("\n" + "="*60)
#     print("GAME SUMMARY")
#     print("="*60)
#     print(f"Total steps: {len(history)}")
#     if history:
#         print(f"Total rounds: {max([e['round'] for e in history]) + 1}")
#     print(f"\nFinal Rewards:")
#     for trader_idx, trader_name in enumerate(['A', 'B', 'C']):
#         print(f"  Trader {trader_name}: {rewards[trader_idx]:.2f}")

#     trades = sum(1 for e in history if e['action'].startswith('trade'))
#     print(f"\nTrades executed: {trades}")
#     print("\nLast 10 events:")
#     for e in history[-10:]:
#         print(f"  Step {e['step']}: {e['trader']} -> {e['action']:15s} | "
#               f"Bid={e['bid_price']:.1f}, Ask={e['ask_price']:.1f}, Spread={e['spread']:.1f} | "
#               f"Reward={e['reward']:.2f}")

# if __name__ == "__main__":
#     print("Pre-compiling Q-function...")
#     Q(0)  # pre-compile Q

#     # Example simulation
#     print("\nRunning game simulation...")
#     agents_mu = [100.0, 101, 99]  # Different beliefs for each trader
#     history, rewards = simulate_game(
#         num_rounds=5,
#         agents_mu=agents_mu,
#         t_horizon=3,
#         beta=2.0,  # Higher beta = more rational
#         seed=42
#     )

#     # Print summary
#     print_game_summary(history, rewards)

#     # Plot results
#     print("\nGenerating visualization...")
#     plot_game_history(history, save_path='game_history.png')

#     # Benchmark
#     import timeit
#     print("\nBenchmarking Q-function...")
#     time = timeit.timeit(
#         lambda: Q.cache_clear() or Q(1).block_until_ready(),
#         number=10
#     )
#     print(f"Q-function benchmark: {time/10:.4f} seconds per call")
