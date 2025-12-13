"""
Memo probabilistic program for Trade or Tighten MDP.
"""
from functools import cache
from memo import memo
import jax
import jax.numpy as np

# example
bids = np.arange(90, 111, 1)
asks = np.arange(90, 111, 1)

# state definition
bid_indices = np.arange(len(bids))
ask_indices= np.arange(len(asks))
turns = np.arange(3)  # each trader takes a turn in order A, B, C: 0 = A, 1 = B, 2 = C

S = np.array(
    np.meshgrid(bid_indices, ask_indices, turns, indexing="ij")
).reshape(3, -1).T

S = S[S[:, 0] < S[:, 1]] # bid < ask

print("Number of states:", len(S))
print("States:", S)
print("Bid indices:", bid_indices)
print("Ask indices:", ask_indices)
print("Turns:", turns)

# action definition
A = np.array([0, 1, 2, 3]) # 0: tighten bid (+1), 1: tighten ask (-1), 2: trade buy, 3: trade sell
# fixed tightening amount: 1

# reward function
@jax.jit
def reward(s, a, mu_i):
    bid_index, ask_index, _ = s

    bid_price = bids[bid_index]
    ask_price = asks[ask_index]

    buy_reward = mu_i - ask_price
    sell_reward = bid_price - mu_i

    return np.where(
        a == 2,  buy_reward, np.where(a == 3, sell_reward, 0.0)
    )

# transition function
@jax.jit
def Tr(s, a, s_):
    s_next = transition_deterministic(s, a)
    return 1.0 * np.all(s_next == s_)

@jax.jit
def transition_deterministic(s, a):
    bid_index, ask_index, turn_index = s  # indices
    bid_index2, ask_index2 = bid_index, ask_index # default
    turn_index2 = (turn_index + 1) % 3
    # tighten bid
    bid_index2 = np.where(
        (a == 0) & ((bid_index + 1) < ask_index),
        bid_index + 1,
        bid_index
    )

    # tighten ask
    ask_index2 = np.where(
        (a == 1) & (bid_index < (ask_index - 1)),
        ask_index - 1,
        ask_index
    )

    # trade: reset spread and rotate starting player
    bid_index2 = np.where(is_trade(a), 0, bid_index2)                 # minimum bid index
    ask_index2 = np.where(is_trade(a), len(asks) - 1, ask_index2)     # maximum ask index
    turn_index2 = np.where(is_trade(a), (turn_index + 1) % 3, turn_index2)

    return np.array([bid_index2, ask_index2, turn_index2])

@jax.jit
def is_trade(a):
    return (a == 2) | (a == 3)

@jax.jit
def is_terminating(s, a):  # current implementation assumes only 1 round
    return is_trade(a)


