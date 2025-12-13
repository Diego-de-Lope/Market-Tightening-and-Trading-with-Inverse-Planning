"""
Memo probabilistic program for Trade or Tighten MDP.
"""
from functools import cache
from memo import memo
import jax
import jax.numpy as np

# example
bids = np.arange(98, 103, 1)
asks = np.arange(98, 103, 1)
# mu_values = [float(m) for m in np.arange(90, 111, 1)]
mu_values = [float(m) for m in np.arange(98, 103, 1)]

# state definition
bid_indices = np.arange(len(bids))
ask_indices= np.arange(len(asks))
turns = np.arange(3)  # each trader takes a turn in order A, B, C: 0 = A, 1 = B, 2 = C

S_array = np.array(
    np.meshgrid(bid_indices, ask_indices, turns, indexing="ij")
).reshape(3, -1).T

S_array = S_array[S_array[:, 0] < S_array[:, 1]] # bid < ask

# Convert to list of tuples for Memo (tuples are hashable, arrays are not)
S = [tuple(map(int, s)) for s in S_array]

print("Number of states:", len(S))
print("First 5 states:", S[:5])
print("Bid indices:", bid_indices)
print("Ask indices:", ask_indices)
print("Turns:", turns)

# action definition
A = [0, 1, 2, 3]  # 0: tighten bid (+1), 1: tighten ask (-1), 2: trade buy, 3: trade sell
# fixed tightening amount: 1

# reward function
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
def Tr(s, a, s_):
    s_next = transition_deterministic(s, a)
    return 1.0 * np.all(s_next == s_)

def transition_deterministic(s, a):

    print(bid_index, ask_index, turn_index)
    bid_index, ask_index, turn_index = s # indices
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

    # Return as tuple for hashability (needed for Memo)
    return (bid_index2, ask_index2, turn_index2)

def is_trade(a):
    return (a == 2) | (a == 3)

def is_terminating(s, a):  # current implementation assumes only 1 round
    return is_trade(a)

@cache
@memo
def Q[s: S, a: A, mu_i: mu_values](t):
    agent: knows(s, a, mu_i)
    agent: given(s_ in S, wpp=Tr(s, a, s_))
    agent: chooses(
        a_ in A,
        to_maximize=(
            0.0 if (t < 0) else
            0.0 if is_terminating(s, a) else
            Q[s_, a_, mu_i](t - 1)
        )
    )
    return E[
        reward(s, a, mu_i)
        + (
            0.0 if (t < 0) else
            0.0 if is_terminating(s, a) else
            Q[agent.s_, agent.a_, mu_i](t - 1)
        )
    ]

if __name__ == "__main__":
    Q(0)  # pre-compile Q
    import timeit
    # Time the operation
    time = timeit.timeit(
        lambda: Q.cache_clear() or Q(1).block_until_ready(),
        number=10
    )
    print(f"Average time: {time/10:.4f} seconds")
