from functools import cache
from memo import memo
import jax
import jax.numpy as np
import numpy as onp


hidden_mu_vals = np.array([98.0, 104.0, 104.0])
belief_mu_vals = np.array([96.0, 98.0, 100.0, 102.0, 104.0])
trader_names = ["A", "B", "C"]
num_turns = 3
TRUE_MU = 100.0

starting_bid = 90.0
starting_ask = 110.0
num_prices = int(starting_ask - starting_bid + 1)
num_mu_states = len(belief_mu_vals)


Prices = np.arange(num_prices)
Turns = np.arange(num_turns)

B = np.arange(num_mu_states)
C = np.arange(num_mu_states)
A = np.array([0, 1, 2]) # 0:Tighten, 1:Buy, 2:Sell


@jax.jit
def get_price_vals(bi, ai):
    return bi + starting_bid, ai + starting_bid

@jax.jit
def is_terminating(a):
    return (a == 1) | (a == 2)

@jax.jit
def get_greedy_action_for_mu(bi, ai, mu_val):
    bid_val, ask_val = get_price_vals(bi, ai)
    can_buy = (mu_val - ask_val) > 0
    can_sell = (bid_val - mu_val) > 0
    can_tighten = (bi + 1) < (ai - 1)
    return np.where(can_buy, 1, np.where(can_sell, 2, np.where(can_tighten, 0, 1)))

@jax.jit
def next_bi_logic(bi, ai, a):
    # If action is 0 (Tighten) and valid, bi increases by 1. Else stays.
    valid_tighten = (bi + 1) < (ai - 1)
    return np.where((a == 0) & valid_tighten, bi + 1, bi)

@jax.jit
def next_ai_logic(bi, ai, a):
    # If action is 0 (Tighten) and valid, ai decreases by 1. Else stays.
    valid_tighten = (bi + 1) < (ai - 1)
    return np.where((a == 0) & valid_tighten, ai - 1, ai)

@jax.jit
def next_ti_logic(ti):
    return (ti + 1) % num_turns

@jax.jit
def R(bi, ai, ti, a):
    bid_val, ask_val = get_price_vals(bi, ai)
    is_hero_turn = (ti == 0)
    is_hero_counterparty = (ti == 1)
    hero_mu = hidden_mu_vals[0]

    active = np.where(a==1, hero_mu - ask_val, np.where(a==2, bid_val - hero_mu, 0.0))
    passive = np.where(a==1, ask_val - hero_mu, np.where(a==2, hero_mu - bid_val, 0.0))
    return np.where(is_hero_turn, active, np.where(is_hero_counterparty, passive, 0.0))

@jax.jit
def update_belief_w(mu_idx, bi, ai, observed_a):
    mu_val = belief_mu_vals[mu_idx]
    pred_a = get_greedy_action_for_mu(bi, ai, mu_val)
    return np.where(pred_a == observed_a, 1.0, 0.001)

@jax.jit
def get_utility(bi_next, ai_next, ti_next, a_cand, future_q_val, b_idx, c_idx):
    is_hero_turn = (ti_next == 0)

    relevant_mu = np.where(ti_next == 1, belief_mu_vals[b_idx],
                  np.where(ti_next == 2, belief_mu_vals[c_idx], 0.0))

    greedy_a = get_greedy_action_for_mu(bi_next, ai_next, relevant_mu)
    is_greedy_move = (a_cand == greedy_a)

    return np.where(
        is_hero_turn, future_q_val, np.where(is_greedy_move, future_q_val, -1.0e9)
    )

@jax.jit
def get_belief_wpp(candidate_idx, current_idx, ti_next, bi_next, ai_next, obs_action, target_turn):
    is_target_turn = (ti_next == target_turn)
    likelihood = update_belief_w(candidate_idx, bi_next, ai_next, obs_action)
    identity = np.where(candidate_idx == current_idx, 1.0, 0.0)
    return np.where(is_target_turn, likelihood, identity)


@memo(cache=True)
def Q[bi: Prices, ai: Prices, ti: Turns, b: B, c: C, a: A](t):
    # Note: State 's' is now split into 'bi, ai, ti'
    agent: knows(bi, ai, ti, b, c, a)
    agent: snapshots_self_as(future_agent)

    return agent [
        R(bi, ai, ti, a) + (0.0 if t <= 0 else imagine [

            future_agent: given(bi_ in Prices, wpp=(bi_ == next_bi_logic(bi, ai, a))),
            future_agent: given(ai_ in Prices, wpp=(ai_ == next_ai_logic(bi, ai, a))),
            future_agent: given(ti_ in Turns,  wpp=(ti_ == next_ti_logic(ti))),


            future_agent: chooses(a_ in A,
                to_maximize=get_utility(
                    bi_, ai_, ti_,
                    a_,
                    Q[bi_, ai_, ti_, b, c, a_](t - 1),
                    b, c
                )
            ),


            future_agent: draws(b_ in B, wpp=get_belief_wpp(b_, b, ti_, bi_, ai_, a_, 1)),
            future_agent: draws(c_ in C, wpp=get_belief_wpp(c_, c, ti_, bi_, ai_, a_, 2)),

            E[ future_agent[ Q[bi_, ai_, ti_, b_, c_, a_](t - 1) ] ]
    ]) ]




def run_simulation(Q_vals):
    # Q_vals shape is now (NumPrices, NumPrices, NumTurns, B, C, A)
    # e.g. (21, 21, 3, 5, 5, 3)

    curr_bid = starting_bid
    curr_ask = starting_ask
    curr_turn = 0
    no_progress_count = 0

    true_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
    hidden_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
    prob_b = np.ones(num_mu_states) / num_mu_states
    prob_c = np.ones(num_mu_states) / num_mu_states

    print(f"\n--- SIMULATION START: Bid {curr_bid} | Ask {curr_ask} ---")

    for step in range(15):
        trader = trader_names[curr_turn]
        prev_trader_idx = (curr_turn - 1) % 3
        prev_trader = trader_names[prev_trader_idx]

        b_idx = int(curr_bid - starting_bid)
        a_idx = int(curr_ask - starting_bid)

        if curr_turn == 0:

            # Index directly using the factored state components
            q_s = Q_vals[b_idx, a_idx, curr_turn]

            # Marginalize beliefs
            q_weighted_c = np.tensordot(prob_c, q_s, axes=([0], [1]))
            q_expected = np.tensordot(prob_b, q_weighted_c, axes=([0], [0]))

            action = int(np.argmax(q_expected))
            top_b = belief_mu_vals[np.argmax(prob_b)]
            top_c = belief_mu_vals[np.argmax(prob_c)]
            source = f"POMDP (Belief Top B: {top_b}, Top C: {top_c})"

        else:
            # --- OPPONENTS ---
            mu_i = hidden_mu_vals[curr_turn]
            action = int(get_greedy_action_for_mu(b_idx, a_idx, mu_i))
            source = f"Greedy (Mu={hidden_mu_vals[curr_turn]})"

            # Update Beliefs
            lik_fn = lambda idx: update_belief_w(idx, b_idx, a_idx, action)
            if curr_turn == 1:
                likelihoods = np.array([lik_fn(i) for i in range(num_mu_states)])
                prob_b = (prob_b * likelihoods) / np.sum(prob_b * likelihoods)
            elif curr_turn == 2:
                likelihoods = np.array([lik_fn(i) for i in range(num_mu_states)])
                prob_c = (prob_c * likelihoods) / np.sum(prob_c * likelihoods)

        act_str = ["Tighten", "BUY", "SELL"][action]
        print(f"Step {step} | {trader} [{source}] decides to {act_str}")

        if action == 1: # BUY
            r_actor = TRUE_MU - curr_ask
            r_counter = curr_ask - TRUE_MU
            r_actor_hidden = float(hidden_mu_vals[curr_turn]) - curr_ask
            r_counter_hidden = curr_ask - float(hidden_mu_vals[prev_trader_idx])
            true_rewards[trader] += r_actor
            true_rewards[prev_trader] += r_counter
            hidden_rewards[trader] += r_actor_hidden
            hidden_rewards[prev_trader] += r_counter_hidden
            print(f" >>> EXECUTION: {trader} BUYS from {prev_trader} @ {curr_ask}")
            break
        elif action == 2: # SELL
            r_actor = curr_bid - TRUE_MU
            r_counter = TRUE_MU - curr_bid
            r_actor_hidden = curr_bid - float(hidden_mu_vals[curr_turn])
            r_counter_hidden = float(hidden_mu_vals[prev_trader_idx]) - curr_bid
            true_rewards[trader] += r_actor
            true_rewards[prev_trader] += r_counter
            hidden_rewards[trader] += r_actor_hidden
            hidden_rewards[prev_trader] += r_counter_hidden
            print(f" >>> EXECUTION: {trader} SELLS to {prev_trader} @ {curr_bid}")
            break
        elif action == 0: # TIGHTEN
            if (curr_bid + 1) < (curr_ask - 1):
                curr_bid += 1; curr_ask -= 1
                print(f"     Update: Spread tightens to {curr_bid} ... {curr_ask}")
            else:
                no_progress_count += 1
                if no_progress_count >= 3:
                    print(f"\n >>> TERMINATION: Market Locked."); break
        curr_turn = (curr_turn + 1) % 3

    print("\n" + "="*60)
    print("FINAL SCORES:", true_rewards)
    print("="*60)
    print("\n" + "="*60)
    print("HIDDEN SCORES:", hidden_rewards)
    print("="*60)


if __name__ == "__main__":
    print("Compiling Policy...")
    # This will now compile MUCH faster
    Q_vals = Q(3).block_until_ready()
    print("Policy compiled. Simulating...")
    run_simulation(Q_vals)
