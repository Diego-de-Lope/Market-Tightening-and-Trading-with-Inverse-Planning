#POMDP VERSION FOR EXPERIMENT #3 (TESTING CONFIDENCE VALUES)


# CONFIDENCE-WEIGHTED POMDP
#imports

from memo import memo
import jax
import jax.numpy as np
import numpy as onp
import time
from tabulate import tabulate

print("\n=== Initializing Configuration ===")

# ----------------------------
# 1) CONFIG
# ----------------------------
hidden_mu_vals = np.array([92.0, 105.0, 107.0])
belief_mu_vals = np.array([92.0, 95.0, 100.0, 105.0, 107.0, 110.0])
num_mu_states = int(belief_mu_vals.shape[0])
ALICE_PRIVATE_PRIOR = 95.0
TRUE_MU = 100.0
BETA_ASSUMED_B = 0.5
BETA_ASSUMED_C = 2.0
starting_bid = 98
starting_ask = 103
num_prices = int(starting_ask - starting_bid + 1)
NUM_ROUNDS = 3
MAX_STEPS = 30
DEPTH = 1

# ----------------------------
# 2) DOMAINS (No Lambda here)
# ----------------------------
Prices = np.arange(num_prices)
Turns  = np.arange(3)
BIdx   = np.arange(num_mu_states)
CIdx   = np.arange(num_mu_states)
Acts   = np.arange(3)

# ----------------------------
# 3) STATIC HELPERS (JIT)
# ----------------------------
@jax.jit
def price_from_index(i):
    return i + starting_bid

@jax.jit
def get_logits(bi, ai, mu_val):
    bid = price_from_index(bi)
    ask = price_from_index(ai)
    u_buy = mu_val - ask
    u_sell = bid - mu_val
    can_tighten = (bi + 1) < (ai - 1)
    u_tighten = np.where(can_tighten, 0.0, -1e9)
    return np.array([u_tighten, u_buy, u_sell])

@jax.jit
def greedy_action(bi, ai, mu_val):
    return np.argmax(get_logits(bi, ai, mu_val))

@jax.jit
def stochastic_action(bi, ai, mu_val, key):
    logits = get_logits(bi, ai, mu_val)
    return jax.random.categorical(key, logits)

@jax.jit
def next_bi(bi, ai, a):
    valid = (bi + 1) < (ai - 1)
    return np.where((a == 0) & valid, bi + 1, bi)

@jax.jit
def next_ai(bi, ai, a):
    valid = (bi + 1) < (ai - 1)
    return np.where((a == 0) & valid, ai - 1, ai)

@jax.jit
def next_turn(t):
    return (t + 1) % 3

@jax.jit
def likelihood_soft(mu_idx, bi, ai, obs_a, beta):
    mu_val = belief_mu_vals[mu_idx]
    logits = get_logits(bi, ai, mu_val)
    probs = jax.nn.softmax(beta * logits)
    return probs[obs_a]

@jax.jit
def belief_wpp(candidate_idx, current_idx, t_next, bi_next, ai_next, obs_a, target_turn):
    is_target = (t_next == target_turn)
    beta = np.where(target_turn == 1, BETA_ASSUMED_B, BETA_ASSUMED_C)
    L = likelihood_soft(candidate_idx, bi_next, ai_next, obs_a, beta)
    identity = np.where(candidate_idx == current_idx, 1.0, 0.0)
    return np.where(is_target, L, identity)

@jax.jit
def get_utility(bi, ai, t, b_idx, c_idx, a_cand, future_val):
    is_hero = (t == 0)
    mu_opp = np.where(t == 1, belief_mu_vals[b_idx], belief_mu_vals[c_idx])
    greedy_a = greedy_action(bi, ai, mu_opp)
    is_valid = is_hero | (a_cand == greedy_a)
    return np.where(is_valid, future_val, -1.0e9)

@jax.jit
def R(bi, ai, t, a, b_idx, c_idx, lam_val):
    is_hero = (t == 0)
    bid = price_from_index(bi)
    ask = price_from_index(ai)

    mu_b = belief_mu_vals[b_idx]
    mu_c = belief_mu_vals[c_idx]
    social_mu = 0.5 * (mu_b + mu_c)

    eff_mu = (1.0 - lam_val) * ALICE_PRIVATE_PRIOR + lam_val * social_mu

    active = np.where(a == 1, eff_mu - ask,
             np.where(a == 2, bid - eff_mu, 0.0))
    return np.where(is_hero, active, 0.0)



# 4) GLOBAL MEMO DEFINITION
@memo(cache=True)
def Q[bi: Prices, ai: Prices, t: Turns, b: BIdx, c: CIdx, a: Acts](depth, lam_val):
    agent: knows(bi, ai, t, b, c, a)
    agent: snapshots_self_as(future)

    return agent[
        R(bi, ai, t, a, b, c, lam_val) + (0.0 if depth <= 0 else imagine[
            future: given(bi_ in Prices, wpp=(bi_ == next_bi(bi, ai, a))),
            future: given(ai_ in Prices, wpp=(ai_ == next_ai(bi, ai, a))),
            future: given(t_  in Turns,  wpp=(t_  == next_turn(t))),

            future: chooses(a_ in Acts, to_maximize=get_utility(
                bi_, ai_, t_, b, c, a_,
                Q[bi_, ai_, t_, b, c, a_](depth - 1, lam_val)
            )),

            future: draws(b_ in BIdx, wpp=belief_wpp(b_, b, t_, bi_, ai_, a_, 1)),
            future: draws(c_ in CIdx, wpp=belief_wpp(c_, c, t_, bi_, ai_, a_, 2)),

            E[ future[ Q[bi_, ai_, t_, b_, c_, a_](depth - 1, lam_val) ] ]
        ])
    ]




# 5) MAIN LOOP

def run_lambda_sweep(lambdas, seed=0):
    results = {}

    for lam_val in lambdas:
        print("\n" + "="*60)
        print(f"Compiling for λ={lam_val:.1f}...")
        t0 = time.time()
        Q_vals = Q(DEPTH, lam_val).block_until_ready()
        print(f"Compiled in {time.time()-t0:.2f}s")

        # Setup Beliefs
        prob_b = onp.ones(num_mu_states) / num_mu_states
        prob_c = onp.ones(num_mu_states) / num_mu_states
        total_pnl = {"A": 0.0, "B": 0.0, "C": 0.0}
        key = jax.random.PRNGKey(seed)
        curr_turn = 0

        # Run 3 Rounds
        for round_num in range(1, NUM_ROUNDS + 1):
            bid, ask = starting_bid, starting_ask
            round_pnl = {"A": 0.0, "B": 0.0, "C": 0.0}
            print(f"--- R{round_num} [λ={lam_val}] ---")

            for step in range(MAX_STEPS):
                bi = int(bid - starting_bid)
                ai = int(ask - starting_bid)

                if curr_turn == 0:
                    # A's Turn: Use Q-Table
                    q_s = Q_vals[bi, ai, curr_turn]
                    q_w_c = onp.tensordot(prob_c, onp.array(q_s), axes=([0],[1]))
                    q_exp = onp.tensordot(prob_b, q_w_c, axes=([0],[0]))

                    if (bid + 1) >= (ask - 1): q_exp[0] = -1e9 
                    action = int(onp.argmax(q_exp))

                    soc_mu = 0.5 * (onp.dot(prob_b, belief_mu_vals) + onp.dot(prob_c, belief_mu_vals))
                    eff_mu = (1.0 - lam_val) * ALICE_PRIVATE_PRIOR + lam_val * soc_mu
                    print(f"  Step {step} | A (Eff μ={eff_mu:.1f}) -> {['Tighten','BUY','SELL'][action]}")

                else:
                    # Opponent Turn
                    mu_true = float(hidden_mu_vals[curr_turn])
                    key, sk = jax.random.split(key)
                    action = int(stochastic_action(bi, ai, mu_true, sk))
                    name = ['A','B','C'][curr_turn]
                    print(f"  Step {step} | {name} (True μ={mu_true:.0f}) -> {['Tighten','BUY','SELL'][action]}")

                    # Update Beliefs
                    lik_fn = lambda i: float(likelihood_soft(i, bi, ai, action, BETA_ASSUMED_B if curr_turn==1 else BETA_ASSUMED_C))
                    L = onp.array([lik_fn(i) for i in range(num_mu_states)])
                    if curr_turn == 1:
                        prob_b = (prob_b * L) / (prob_b * L).sum()
                    else:
                        prob_c = (prob_c * L) / (prob_c * L).sum()

                # Execution Logic
                if action == 1: # BUY
                    p = ask
                    pnl_act = TRUE_MU - p
                    pnl_pas = p - TRUE_MU
                    name_act = ['A','B','C'][curr_turn]
                    name_pas = ['A','B','C'][(curr_turn-1)%3]
                    round_pnl[name_act] += pnl_act
                    round_pnl[name_pas] += pnl_pas
                    print(f"  >>> TRADE: {name_act} BUYS @ {p}")
                    break
                elif action == 2: # SELL
                    p = bid
                    pnl_act = p - TRUE_MU
                    pnl_pas = TRUE_MU - p
                    name_act = ['A','B','C'][curr_turn]
                    name_pas = ['A','B','C'][(curr_turn-1)%3]
                    round_pnl[name_act] += pnl_act
                    round_pnl[name_pas] += pnl_pas
                    print(f"  >>> TRADE: {name_act} SELLS @ {p}")
                    break
                elif action == 0:
                    if (bid+1) < (ask-1):
                        bid += 1; ask -= 1
                    else:
                        break # Locked

                curr_turn = (curr_turn + 1) % 3

            curr_turn = (curr_turn + 1) % 3 # Next round starter
            for k in total_pnl: total_pnl[k] += round_pnl[k]

        results[lam_val] = total_pnl
        print(f"Result λ={lam_val}: {total_pnl}")

    # TABLE
    table = [[f"{l}", f"{r['A']:.2f}", f"{r['B']:.2f}", f"{r['C']:.2f}"] for l, r in results.items()]
    print("\n" + tabulate(table, headers=["Lambda", "A PnL", "B PnL", "C PnL"], tablefmt="github"))






#running

if __name__ == "__main__":
    lambdas = [0.0, 0.5, 1.0]
    run_lambda_sweep(lambdas, seed=42)







#PLOTTING SAVED RESULTS AFTER
#MANUALLY ENCODED SAVED RESULTS FOR THESE GRAPHS FOR SIMPLICITY

import matplotlib.pyplot as plt
import numpy as np

# --- 1. DATA FROM YOUR SIMULATION ---
lambdas = [0.0, 0.5, 1.0]

# PnL Results
pnl_a = [1.0, 4.0, -1.0]   # The "Goldilocks" Curve
pnl_b = [0.0, -1.0, 1.0]
pnl_c = [-1.0, -3.0, 0.0]

# Action Counts for Agent A (Extracted from your logs)
actions_sell =    [1, 0, 0]
actions_tighten = [0, 2, 1]
actions_buy =     [0, 0, 1]

# --- 2. PLOTTING SETUP ---
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'}) # Academic Style
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5)) # Slightly wider for better spacing

# --- PLOT 1: PERFORMANCE CURVE (The Goldilocks Zone) ---
# Plot opponents faintly for context
ax1.plot(lambdas, pnl_b, 'o--', color='gray', alpha=0.3, label='Opponents (B/C)')
ax1.plot(lambdas, pnl_c, 'o--', color='gray', alpha=0.3)

# Plot Agent A boldly
ax1.plot(lambdas, pnl_a, 'o-', color='blue', linewidth=3, markersize=10, label='Agent A (Hero)')

# --- FIXED ANNOTATIONS ---

# 1. Risk Averse: Align LEFT so it starts at the axis, doesn't cross it
ax1.annotate('Risk Averse\n(Sells Low)',
             xy=(0.0, 1.0),             # Point to the data
             xytext=(0.02, 2.5),        # Text slightly offset to right
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             ha='left',                 # Left alignment prevents y-axis overlap
             fontsize=10)

# 2. Optimal: Keep Center
ax1.annotate('Optimal\n(Market Maker)',
             xy=(0.5, 4.0),
             xytext=(0.5, 2.8),         # Moved text down slightly for cleaner look
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             ha='center',
             fontsize=10)

# 3. Herding: Align RIGHT so it ends at the axis, shifted slightly right
ax1.annotate('Herding\n(Buys High)',
             xy=(1.0, -1.0),
             xytext=(1.02, 0.5),        # Text slightly offset to right
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             ha='right',                # Right alignment prevents border cutoff
             fontsize=10)

ax1.set_title('Impact of Social Cognition (λ) on Profit')
ax1.set_xlabel('Lambda (λ) - Social Reliance')
ax1.set_ylabel('Total PnL ($)')
ax1.set_xticks(lambdas)
ax1.set_ylim(-4, 5)
ax1.axhline(0, color='black', linewidth=0.5)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='lower left')

# --- PLOT 2: BEHAVIORAL SHIFT (Stacked Bar Chart) ---
bar_width = 0.3
indices = np.arange(len(lambdas))

# Create stacked bars
p1 = ax2.bar(indices, actions_sell, bar_width, label='SELL', color='#d62728', alpha=0.8)
p2 = ax2.bar(indices, actions_tighten, bar_width, bottom=actions_sell, label='TIGHTEN', color='#7f7f7f', alpha=0.8)
bottom_buy = np.array(actions_sell) + np.array(actions_tighten)
p3 = ax2.bar(indices, actions_buy, bar_width, bottom=bottom_buy, label='BUY', color='#2ca02c', alpha=0.8)

ax2.set_title("Agent A's Strategy Composition")
ax2.set_xlabel('Lambda (λ)')
ax2.set_ylabel('Count of Actions')
ax2.set_xticks(indices)
ax2.set_xticklabels([str(l) for l in lambdas])
ax2.set_yticks([0, 1, 2, 3])
ax2.legend(loc='upper right') # Moved legend to avoid covering bars
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

# --- SAVE & SHOW ---
plt.tight_layout()
plt.savefig('social_cognition_results_fixed.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'social_cognition_results_fixed.png'")
plt.show()