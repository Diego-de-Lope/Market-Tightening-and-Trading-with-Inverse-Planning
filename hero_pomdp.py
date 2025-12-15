from functools import cache
from memo import memo
import jax
import jax.numpy as np
import numpy as onp

hidden_mu_vals = np.array([98.0, 104.0, 104.0]) # A, B, C
belief_mu_vals = np.array([96.0, 98.0, 100.0, 102.0, 104.0]) # A, B, C
trader_names = ["A", "B", "C"]
num_turns = 3

# True market value of the asset (used for reward calculation)
TRUE_MU = 100.0

# Initial Market State
starting_bid = 90.0
starting_ask = 110.0
num_prices = int(starting_ask - starting_bid + 1)
num_mu_states = len(belief_mu_vals)

# State Space: Bid_Index * Ask_Index * Turn_Index
S = np.arange(num_prices * num_prices * num_turns * num_mu_states)

# Actions: 0: tighten bid (+1) & ask (-1), 1: trade buy, 2: trade sell
A = np.array([0, 1, 2])


@jax.jit
def decode(s):
    """
    Decodes state index into (bid, ask, turn, mu_b_idx, mu_c_idx).
    """
    mu_c_idx = s % num_mu_states
    s = s // num_mu_states

    mu_b_idx = s % num_mu_states
    s = s // num_mu_states

    bi = s % num_prices
    ai = (s // num_prices) % num_prices
    ti = s // (num_prices * num_prices)

    return bi, ai, ti, mu_b_idx, mu_c_idx

@jax.jit
def encode(bi, ai, ti, mu_b_idx, mu_c_idx):
    """
    Encodes components into single state index.
    """
    s = ti
    s = s * num_prices + ai
    s = s * num_prices + bi
    s = s * num_mu_states + mu_b_idx
    s = s * num_mu_states + mu_c_idx
    return s

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
    _, _, ti_next, _, _ = decode(s_next)
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
def get_greedy_action_for_mu(bi, ai, mu_val):
    """
    Predicts what a trader with valuation 'mu_val' would do.
    """
    bid_val, ask_val = get_price_vals(bi, ai)

    can_buy = (mu_val - ask_val) > 0
    can_sell = (bid_val - mu_val) > 0
    can_tighten = (bi + 1) < (ai - 1)

    return np.where(can_buy, 1,
           np.where(can_sell, 2,
           np.where(can_tighten, 0, 1)))


@jax.jit
def get_greedy_action_jax(s):
    """
    Extracts the correct opponent Mu from the state and predicts action.
    """
    bi, ai, ti, mu_b_idx, mu_c_idx = decode(s)

    current_mu_idx = np.where(ti == 1, mu_b_idx,
                     np.where(ti == 2, mu_c_idx, 0))

    current_mu_val = belief_mu_vals[current_mu_idx]

    return get_greedy_action_for_mu(bi, ai, current_mu_val)

@jax.jit
def Tr(s, a, s_):
    """
    Transition Function P(s' | s, a).
    """
    bi, ai, ti, mb, mc = decode(s)
    ti_next = (ti + 1) % num_turns
    valid_tighten = (bi + 1) < (ai - 1)
    bi_next = np.where((a == 0) & valid_tighten, bi + 1, bi)
    ai_next = np.where((a == 0) & valid_tighten, ai - 1, ai)
    expected_s_ = encode(bi_next, ai_next, ti_next, mb, mc)
    return 1.0 * (s_ == expected_s_)

@jax.jit
def R(s, a):
    """
    Reward logic for the PLANNER (Hero/Agent A).
    Includes both ACTIVE trading profit and PASSIVE counterparty profit.
    """
    bi, ai, ti, _, _ = decode(s)
    bid_val, ask_val = get_price_vals(bi, ai)

    is_hero_turn = (ti == 0)
    is_hero_counterparty = (ti == 1)
    hero_mu = hidden_mu_vals[0]

    hero_active_r = np.where(a == 1, hero_mu - ask_val,
                    np.where(a == 2, bid_val - hero_mu,
                    0.0))

    hero_passive_r = np.where(a == 1, ask_val - hero_mu,
                     np.where(a == 2, hero_mu - bid_val,
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

@jax.jit
def update_belief(belief_vector, bid_idx, ask_idx, observed_action):
    """
    Vectorized Bayesian Update.
    Computes likelihoods for ALL possible Mu values in parallel.
    """
    # 1. Run the greedy predictor on ALL possible Mus at once
    # in_axes=(None, None, 0) means:
    #   - Keep bid_idx constant (None)
    #   - Keep ask_idx constant (None)
    #   - Map over the 0-th dimension of POSSIBLE_MUS
    all_pred_actions = jax.vmap(get_greedy_action_for_mu, in_axes=(None, None, 0))(
        bid_idx, ask_idx, belief_mu_vals
    )

    # 2. Compare predictions vs reality (Vectorized)
    # If prediction matches observed_action -> Likelihood 1.0
    # If mismatch -> Likelihood 0.001 (epsilon)
    likelihoods = np.where(all_pred_actions == observed_action, 1.0, 0.001)

    # 3. Bayes Update (Prior * Likelihood)
    unnormalized_posterior = belief_vector * likelihoods

    # 4. Normalize
    total_prob = np.sum(unnormalized_posterior)

    # Return normalized (safe divide to avoid NaN if total is 0)
    return np.where(total_prob > 0,
                     unnormalized_posterior / total_prob,
                     belief_vector)

def get_expected_q(Q_vals, current_bi, current_ai, current_turn, belief_b, belief_c):
    """
    Calculates E[Q] by summing Q-values weighted by the probability of being in that world.
    """
    expected_q_values = np.zeros(3) # For actions 0, 1, 2

    # Marginalize over all possible worlds (combinations of Mu_B and Mu_C)
    for b_idx in range(num_mu_states):
        prob_b = belief_b[b_idx]
        if prob_b < 0.001: continue

        for c_idx in range(num_mu_states):
            prob_c = belief_c[c_idx]
            if prob_c < 0.001: continue

            # The probability of this specific world
            joint_prob = prob_b * prob_c

            # Get the state index for this world
            s_idx = encode(current_bi, current_ai, current_turn, b_idx, c_idx)

            # Add weighted Q-values
            expected_q_values += joint_prob * Q_vals[s_idx, :]

    return expected_q_values

def run_simulation(Q_vals):
    curr_bid = starting_bid
    curr_ask = starting_ask
    curr_turn = 0 # Start with Agent A
    no_progress_count = 0
    belief_b = np.ones(num_mu_states) / num_mu_states
    belief_c = np.ones(num_mu_states) / num_mu_states

    true_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
    hidden_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}

    print(f"\n--- SIMULATION START: Bid {curr_bid} | Ask {curr_ask} | True Value {TRUE_MU} ---")

    for step in range(15):
        trader = trader_names[curr_turn]

        prev_trader_idx = (curr_turn - 1) % 3
        prev_trader = trader_names[prev_trader_idx]

        b_idx = int(curr_bid - starting_bid)
        a_idx = int(curr_ask - starting_bid)

        if curr_turn == 0:
            # --- AGENT A (HERO) uses Planned Policy ---
            q_expected = get_expected_q(Q_vals, b_idx, a_idx, curr_turn, belief_b, belief_c)
            action = int(np.argmax(q_expected))
            source = "POMDP (Belief-Based)"
        else:
            # --- AGENTS B/C use Greedy Policy ---
            mu_i = hidden_mu_vals[curr_turn]
            action = int(get_greedy_action_for_mu(b_idx, a_idx, mu_i))
            source = f"Greedy (Mu={hidden_mu_vals[curr_turn]})"

        # Log the attempt
        act_str = ["Tighten", "BUY", "SELL"][action]
        print(f"Step {step} | {trader} [{source}] decides to {act_str}")

        # If an opponent acted, Agent A learns from it
        if curr_turn == 1: # B acted
            belief_b = update_belief(belief_b, b_idx, a_idx, action)
            # Print top hypothesis
            top_b = belief_mu_vals[np.argmax(belief_b)]
            print(f"     [A's Belief] Thinks B is likely: {top_b} (Prob: {np.max(belief_b):.2f})")

        elif curr_turn == 2: # C acted
            belief_c = update_belief(belief_c, b_idx, a_idx, action)
            top_c = belief_mu_vals[np.argmax(belief_c)]
            print(f"     [A's Belief] Thinks C is likely: {top_c} (Prob: {np.max(belief_c):.2f})")

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

if __name__ == "__main__":
    print("Compiling Policy...")
    Q_vals = Q(10).block_until_ready()
    run_simulation(Q_vals)
