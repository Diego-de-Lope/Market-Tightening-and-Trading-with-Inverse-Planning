import jax
import jax.numpy as jnp
from memo import memo
import matplotlib.pyplot as plt
import numpy as np

# Set a seed for reproducibility of stochastic actions
KEY = jax.random.PRNGKey(42)

# --- 1. DOMAINS ---
PRICE_D = jnp.arange(11)        # Prices 95-105
TRADER_D = jnp.array([0, 1, 2]) # 0: Alice, 1: Bob, 2: Charlie
ACT_D = jnp.array([0, 1, 2, 3]) # 0: Sell, 1: Imp.Bid, 2: Imp.Ask, 3: Buy
MU_D = jnp.array([99.0, 100.0, 101.0])

# --- JAX HELPERS & Q-FUNCTION (Omitting for brevity, using original functions) ---
# ... (All helper functions and the Q-function definition are here) ...

@jax.jit
def get_next_bi(bi, ai, a):
    return jnp.where((a == 1) & (bi + 1 < ai), bi + 1, bi)

@jax.jit
def get_next_ai(bi, ai, a):
    return jnp.where((a == 2) & (bi < ai - 1), ai - 1, ai)

@jax.jit
def get_reward(bi, ai, bp, ap, turn, a, mu_a):
    b_val, a_val = bi + 95.0, ai + 95.0
    active_r = jnp.where(a == 0, b_val - mu_a, jnp.where(a == 3, mu_a - a_val, 0.0))
    passive_r = jnp.where((a == 0) & (bp == 0), mu_a - b_val,
                 jnp.where((a == 3) & (ap == 0), a_val - mu_a, 0.0))
    return jnp.where(turn == 0, active_r, passive_r)

@jax.jit
def get_greedy_action(bi, ai, mu):
    b_val, a_val = bi + 95.0, ai + 95.0
    can_buy = (mu - a_val > 0.1)
    can_sell = (b_val - mu > 0.1)
    bid_dist = mu - b_val
    ask_dist = a_val - mu
    should_improve_bid = bid_dist > ask_dist
    improve_action = jnp.where(should_improve_bid, 1, 2)
    should_improve = (bi < ai - 1)
    return jnp.where(can_buy, 3,
           jnp.where(can_sell, 0,
           jnp.where(should_improve, improve_action, 0)))

@jax.jit
def get_utility(bi, ai, bp, ap, turn, mb, mc, a_cand, future_q):
    mu_peer = jnp.where(turn == 1, mb, mc)
    greedy_a = get_greedy_action(bi, ai, mu_peer)
    return jnp.where((turn == 0) | (a_cand == greedy_a), future_q, -1e6)

@jax.jit
def get_likelihood(bi, ai, mu_hypothesized, action_observed, noise_prob):
    rational_a = get_greedy_action(bi, ai, mu_hypothesized)
    is_rational_a = (action_observed == rational_a)
    L_rational = (1.0 - noise_prob) + (noise_prob / 4.0)
    L_random = (noise_prob / 4.0)
    return jnp.where(is_rational_a, L_rational, L_random)


@memo(cache=True)
def Q[bi: PRICE_D, ai: PRICE_D, bp: TRADER_D, ap: TRADER_D, turn: TRADER_D,
      mb: MU_D, mc: MU_D, a: ACT_D](t, mu_a=100.0):
    alice: knows(bi, ai, bp, ap, turn, mb, mc, a)
    alice: snapshots_self_as(f)

    return alice [
        get_reward(bi, ai, bp, ap, turn, a, mu_a) + (
            0.0 if t <= 0 else 0.9 * ((a == 1) + (a == 2)) * imagine [
                f: given(bi_ in PRICE_D, wpp=(bi_ == get_next_bi(bi, ai, a))),
                f: given(ai_ in PRICE_D, wpp=(ai_ == get_next_ai(bi, ai, a))),
                f: given(bp_ in TRADER_D, wpp=((a == 1) * (bp_ == turn)) + ((a != 1) * (bp_ == bp))),
                f: given(ap_ in TRADER_D, wpp=((a == 2) * (ap_ == turn)) + ((a != 2) * (ap_ == ap))),
                f: given(turn_ in TRADER_D, wpp=(turn_ == (turn + 1) % 3)),
                f: chooses(a_next in ACT_D,
                    to_maximize=get_utility(bi_, ai_, bp_, ap_, turn_, mb, mc, a_next,
                                            Q[bi_, ai_, bp_, ap_, turn_, mb, mc, a_next](t-1))),
                f: draws(mb_ in MU_D, wpp=(mb_ == mb) * (get_greedy_action(bi_, ai_, mb) == a_next)),
                f: draws(mc_ in MU_D, wpp=(mc_ == mc) * (get_greedy_action(bi_, ai_, mc) == a_next)),
                E[ f [ Q[bi_, ai_, bp_, ap_, turn_, mb_, mc_, a_next](t-1) ] ]
            ]
        )
    ]

# --- 5. EXECUTION & STOCHASTIC SIMULATION ---
print("Compiling Stochastic Asymmetric Agent Model (t=2)...")
res = Q(2).block_until_ready()

# Randomness parameters
NOISE_PROB_BOB = 0.10 # 10% chance Bob is random
NOISE_PROB_CHARLIE = 0.25 # 25% chance Charlie is random
NUM_ACTIONS = len(ACT_D)

# Scenario: Alice (100), Bob (101), Charlie (99)
actual_mu = [100.0, 101.0, 99.0]
bi, ai, bp, ap, turn = 2, 8, 0, 0, 0
history = []
beliefs = [jnp.ones(len(MU_D))/len(MU_D), jnp.ones(len(MU_D))/len(MU_D)] # Beliefs about Bob's mu, Charlie's mu

for s in range(30):
    KEY, subkey = jax.random.split(KEY)

    if turn == 0:
        q_slice = res[bi, ai, bp, ap, turn]
        prob_matrix = jnp.outer(beliefs[0], beliefs[1])
        exp_q = jnp.einsum('bc, bca -> a', prob_matrix, q_slice)
        act_optimal = int(jnp.argmax(exp_q))
        act = act_optimal

    else:
        agent_index = turn - 1
        mu_actual = actual_mu[turn]
        noise_prob = NOISE_PROB_BOB if turn == 1 else NOISE_PROB_CHARLIE

        rational_a = get_greedy_action(bi, ai, mu_actual)
        random_choice = jax.random.choice(subkey, ACT_D)
        is_random = jax.random.uniform(subkey) < noise_prob

        act_optimal = int(rational_a) # Keep track of the rational action
        act = int(jnp.where(is_random, random_choice, rational_a))

    pnl = float(get_reward(bi, ai, bp, ap, turn, act, 100.0))

    if turn > 0:
        agent_index = turn - 1
        noise_prob = NOISE_PROB_BOB if turn == 1 else NOISE_PROB_CHARLIE
        likelihoods = jax.vmap(lambda m: get_likelihood(bi, ai, m, act, noise_prob))(MU_D)
        beliefs[agent_index] = (beliefs[agent_index] * likelihoods)
        beliefs[agent_index] = beliefs[agent_index] / jnp.sum(beliefs[agent_index])

    history.append({'s': s, 'b': bi+95, 'a': ai+95, 'bp': bp, 'ap': ap, 't': turn, 'act': act,
                    'act_opt': act_optimal, 'pnl': pnl,
                    'b_p_101': float(beliefs[0][2]), 'c_p_99': float(beliefs[1][0])})

    # State Transition
    if act == 0 or act == 3:
        bi, ai, bp, ap = 2, 8, 0, 0
    else:
        bi, ai = int(get_next_bi(bi, ai, act)), int(get_next_ai(bi, ai, act))
        if act == 1: bp = turn
        if act == 2: ap = turn
    turn = (turn + 1) % 3

# --- 6. TABLE PRINTING ---
names = ['Alice', 'Bob', 'Charlie']
action_map = {0: 'Sell (0)', 1: '+Bid (1)', 2: '+Ask (2)', 3: 'Buy (3)'}

print("\n## 📊 Simulation Play-by-Play Table (30 Turns)")
print(f"> Alice (µ=100) | Bob (µ=101, 10% Noise) | Charlie (µ=99, 25% Noise)")
print("-" * 115)
header = "| **Turn** | **Player** | **Bid** | **Ask** | **Action (Code)** | **Optimal?** | **Alice PnL** | **P(Bob=101)** | **P(Charlie=99)** |"
print(header)
print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

for h in history:
    player_name = names[h['t']]
    action_str = action_map[h['act']]

    # Check if the executed action was the optimal (rational) one
    if h['t'] == 0:
        optimal_check = "**Optimal**"
    else:
        optimal_check = "Optimal" if h['act'] == h['act_opt'] else f"**MISTAKE** ({action_map[h['act_opt']]})"

    pnl_str = f"{h['pnl']:+.2f}" if h['pnl'] != 0 else "---"

    row = (
        f"| {h['s']:<6} | **{player_name}** | {h['b']:<5} | {h['a']:<5} | {action_str:<15} | "
        f"{optimal_check:<12} | {pnl_str:<11} | {h['b_p_101']:.3f} | {h['c_p_99']:.3f} |"
    )
    print(row)

if len(history) < 30:
    print("\nSimulation ended early due to trade.")
