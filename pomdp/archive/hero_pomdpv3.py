from functools import cache
from memo import memo
import jax
import jax.numpy as np
import numpy as onp

hidden_mu_vals = np.array([95.0, 105.0, 100.0]) # A, B, C
belief_mu_vals = np.array([95.0, 90.0, 100.0, 105.0, 110.0]) # A, B, C
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
S = np.arange(num_prices * num_prices * num_turns)

B = np.arange(num_mu_states)
C = np.arange(num_mu_states)

# Actions: 0: tighten bid (+1) & ask (-1), 1: trade buy, 2: trade sell
A = np.array([0, 1, 2])


@jax.jit
def decode(s):
    bi = s % num_prices
    ai = (s // num_prices) % num_prices
    ti = s // (num_prices * num_prices)
    return bi, ai, ti

@jax.jit
def encode(bi, ai, ti):
    return bi + (ai * num_prices) + (ti * num_prices * num_prices)

@jax.jit
def get_price_vals(bi, ai):
    """Converts 0-based indices back to float prices"""
    bid_val = bi + starting_bid
    ask_val = ai + starting_bid
    return bid_val, ask_val

@jax.jit
def get_utility(s_next, a_candidate, future_q_val, b_idx, c_idx):
    """
    Revised: Accepts belief indices b and c directly.
    Determines relevant Mu internally.
    """
    bi, ai, ti_next = decode(s_next)
    is_hero_turn = (ti_next == 0)

    relevant_mu = np.where(ti_next == 1, belief_mu_vals[b_idx],
                  np.where(ti_next == 2, belief_mu_vals[c_idx], 0.0))

    greedy_a = get_greedy_action_for_mu(bi, ai, relevant_mu)
    is_greedy_move = (a_candidate == greedy_a)

    return np.where(
        is_hero_turn,
        future_q_val,
        np.where(is_greedy_move, future_q_val, -1.0e9)
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

    hero_passive_r = np.where(a == 1, ask_val - hero_mu,
                     np.where(a == 2, hero_mu - bid_val,
                     0.0))

    return np.where(is_hero_turn, hero_active_r,
           np.where(is_hero_counterparty, hero_passive_r,
           0.0))

@jax.jit
def update_belief_w(mu_idx, s, observed_a):
    """
    Returns Likelihood: P(observed_a | mu_idx)
    Used by 'wpp' to weight the future branches.
    """
    bi, ai, _ = decode(s)
    mu_val = belief_mu_vals[mu_idx]
    pred_a = get_greedy_action_for_mu(bi, ai, mu_val)
    return np.where(pred_a == observed_a, 1.0, 0.001)

@jax.jit
def get_belief_wpp(candidate_idx, current_idx, s_next, obs_action, target_turn):
    """
    JAX-compatible switch for belief updates.
    Returns likelihood if it's the target's turn, otherwise returns identity (1.0 or 0.0).
    """
    _, _, ti_next = decode(s_next)
    is_target_turn = (ti_next == target_turn)
    likelihood = update_belief_w(candidate_idx, s_next, obs_action)
    identity = np.where(candidate_idx == current_idx, 1.0, 0.0)
    return np.where(is_target_turn, likelihood, identity)


@memo(cache=True)
def Q[s:S, b: B, c: C, a: A](t):
    agent: knows(s, b, c, a)
    agent: snapshots_self_as(future_agent)
    return agent [ R(s, a) + (0.0 if t <= 0 else imagine [
        future_agent: given(s_ in S, wpp=Tr(s, a, s_)),
        future_agent: chooses(a_ in A,
                to_maximize=get_utility(s_, a_, Q[s_, b, c, a_](t - 1), b, c)
            ),
        future_agent: draws(b_ in B, wpp=get_belief_wpp(b_, b, s_, a_, 1)),
        future_agent: draws(c_ in C, wpp=get_belief_wpp(c_, c, s_, a_, 2)),
        E[ future_agent[ Q[s_, b_, c_, a_](t - 1) ] ]
    ]) ]


def run_simulation(Q_vals):
    curr_bid = starting_bid
    curr_ask = starting_ask
    curr_turn = 0 # Start with Agent A
    no_progress_count = 0

    true_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
    hidden_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
    prob_b = np.ones(num_mu_states) / num_mu_states
    prob_c = np.ones(num_mu_states) / num_mu_states

    print(f"\n--- SIMULATION START: Bid {curr_bid} | Ask {curr_ask} | True Value {TRUE_MU} ---")

    for step in range(15):
        trader = trader_names[curr_turn]

        prev_trader_idx = (curr_turn - 1) % 3
        prev_trader = trader_names[prev_trader_idx]

        b_idx = int(curr_bid - starting_bid)
        a_idx = int(curr_ask - starting_bid)
        s_idx = encode(b_idx, a_idx, curr_turn)

        if curr_turn == 0:

            q_s = Q_vals[s_idx]
            q_weighted_c = np.tensordot(prob_c, q_s, axes=([0], [1]))
            q_expected = np.tensordot(prob_b, q_weighted_c, axes=([0], [0]))
            action = int(np.argmax(q_expected))
            top_b = belief_mu_vals[np.argmax(prob_b)]
            top_c = belief_mu_vals[np.argmax(prob_c)]
            source = f"POMDP (Belief Top B: {top_b}, Top C: {top_c})"
        else:
            mu_i = hidden_mu_vals[curr_turn]
            action = int(get_greedy_action_for_mu(b_idx, a_idx, mu_i))
            source = f"Greedy (Mu={hidden_mu_vals[curr_turn]})"
            lik_fn = lambda idx: update_belief_w(idx, s_idx, action)
            if curr_turn == 1: # B acted
                likelihoods = np.array([lik_fn(i) for i in range(num_mu_states)])
                prob_b = (prob_b * likelihoods) / np.sum(prob_b * likelihoods)
            elif curr_turn == 2: # C acted
                likelihoods = np.array([lik_fn(i) for i in range(num_mu_states)])
                prob_c = (prob_c * likelihoods) / np.sum(prob_c * likelihoods)

        # Log the attempt
        act_str = ["Tighten", "BUY", "SELL"][action]
        print(f"Step {step} | {trader} [{source}] decides to {act_str}")


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
    Q_vals = Q(3).block_until_ready()
    print("Policy compiled successfully. Simulating...")
    run_simulation(Q_vals)
