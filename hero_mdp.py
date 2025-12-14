from functools import cache
from memo import memo
import jax
import jax.numpy as np
import numpy as onp

hidden_mu_vals = np.array([105.0, 95.0, 100.0]) # A, B, C
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

    can_tighten = (bi + 1) < (ai - 1)

    return np.where(can_buy, 1,
           np.where(can_sell, 2,
           np.where(can_tighten, 0,
           1)))

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

def run_simulation(Q_vals):
    curr_bid = starting_bid
    curr_ask = starting_ask
    curr_turn = 0 # Start with Agent A

    rewards = {"A": 0.0, "B": 0.0, "C": 0.0}

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

        # 3. EXECUTE & DISTRIBUTE REWARDS
        if action == 1: # BUY
            # Trader Buys @ Ask from Counterparty
            r_actor = TRUE_MU - curr_ask
            r_counter = curr_ask - TRUE_MU

            rewards[trader] += r_actor
            rewards[prev_trader] += r_counter

            print(f" >>> EXECUTION: {trader} BUYS from {prev_trader} @ {curr_ask}")
            print(f"     {trader} Profit: {r_actor} (TrueMu {TRUE_MU} - Ask {curr_ask})")
            print(f"     {prev_trader} Profit: {r_counter} (Ask {curr_ask} - TrueMu {TRUE_MU})")
            break # Round Ends

        elif action == 2: # SELL
            # Trader Sells @ Bid to Counterparty
            r_actor = curr_bid - TRUE_MU
            r_counter = TRUE_MU - curr_bid

            rewards[trader] += r_actor
            rewards[prev_trader] += r_counter

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
                print(f"     Action 0 (Tighten) chosen but spread is too tight. Prices hold.")

        curr_turn = (curr_turn + 1) % 3

    print("\n" + "="*40)
    print("FINAL SCORES (Based on True Value)")
    print(rewards)
    print("="*40)

if __name__ == "__main__":
    print("Compiling Policy...")
    Q_vals = Q(10).block_until_ready()
    run_simulation(Q_vals)
