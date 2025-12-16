# POMDP VERSION #1 -- 2 PLAYERS

import jax
import jax.numpy as jnp
from memo import memo
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

KEY = jax.random.PRNGKey(42)

# --- 1. DOMAINS ---
PRICE_D = jnp.arange(11)        
TRADER_D = jnp.array([0, 1])    
ACT_D = jnp.array([0, 1, 2, 3]) 
MU_D = jnp.array([99.0, 100.0, 101.0]) 

# --- 2. JAX HELPERS ---
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
    if_trade = jnp.where(can_buy, 3, jnp.where(can_sell, 0, -1))
    bid_dist = mu - b_val
    ask_dist = a_val - mu
    should_improve_bid = bid_dist > ask_dist
    improve_action = jnp.where(should_improve_bid, 1, 2)
    should_improve = (bi < ai - 1)
    return jnp.where(if_trade != -1, if_trade, jnp.where(should_improve, improve_action, 0))

@jax.jit
def get_utility(bi, ai, bp, ap, turn, mb, a_cand, future_q):
    mu_peer = mb
    greedy_a = get_greedy_action(bi, ai, mu_peer)
    return jnp.where((turn == 0) | (a_cand == greedy_a), future_q, -1e6)

@jax.jit
def get_likelihood(bi, ai, mu_hypothesized, action_observed, noise_prob):
    rational_a = get_greedy_action(bi, ai, mu_hypothesized)
    is_rational_a = (action_observed == rational_a)
    L_rational = (1.0 - noise_prob) + (noise_prob / 4.0)
    L_random = (noise_prob / 4.0)
    return jnp.where(is_rational_a, L_rational, L_random)

# --- 3. THE Q-FUNCTION MODEL ---
@memo(cache=True)
def Q[bi: PRICE_D, ai: PRICE_D, bp: TRADER_D, ap: TRADER_D, turn: TRADER_D,
      mb: MU_D, a: ACT_D](t, mu_a=98.0): 
    
    alice: knows(bi, ai, bp, ap, turn, mb, a)
    alice: snapshots_self_as(f)
    
    return alice [
        get_reward(bi, ai, bp, ap, turn, a, mu_a) + (
            0.0 if t <= 0 else 0.95 * ((a == 1) + (a == 2)) * imagine [ 
                f: given(bi_ in PRICE_D, wpp=(bi_ == get_next_bi(bi, ai, a))),
                f: given(ai_ in PRICE_D, wpp=(ai_ == get_next_ai(bi, ai, a))),
                f: given(bp_ in TRADER_D, wpp=((a == 1) * (bp_ == turn)) + ((a != 1) * (bp_ == bp))),
                f: given(ap_ in TRADER_D, wpp=((a == 2) * (ap_ == turn)) + ((a != 2) * (ap_ == ap))),
                f: given(turn_ in TRADER_D, wpp=(turn_ == (turn + 1) % 2)),
                
                f: chooses(a_next in ACT_D, 
                    to_maximize=get_utility(bi_, ai_, bp_, ap_, turn_, mb, a_next, 
                                            Q[bi_, ai_, bp_, ap_, turn_, mb, a_next](t-1))),
                
                f: draws(mb_ in MU_D, wpp=(mb_ == mb) * (get_greedy_action(bi_, ai_, mb) == a_next)),
                
                E[ f [ Q[bi_, ai_, bp_, ap_, turn_, mb, a_next](t-1) ] ]
            ]
        )
    ]

# --- 4. EXECUTION & Q-TABLE EXPORT ---
T_MAX = 6 
print("==============================================================")
print(f"### Starting DP Compilation for T={T_MAX}, Gamma=0.95 (Final Tradeoff Test) ###")
start_time = time.time()

RES_T_MAX = Q(T_MAX).block_until_ready()

action_labels = ['Sell (0)', '+Bid (1)', '+Ask (2)', 'Buy (3)']

def qtable_slice_to_df(res_tensor, bi, ai, bp, ap, turn, mu_domain, act_labels):
    q_slice = np.array(res_tensor[bi, ai, bp, ap, turn]) 
    row_labels = [f'{float(m):.1f}' for m in np.array(mu_domain)]
    df = pd.DataFrame(q_slice, index=row_labels, columns=act_labels)
    df.index.name = r'$B\_\mathrm{Val}$'
    return df

demo_bi, demo_ai, demo_bp, demo_ap, demo_turn = 3, 8, 0, 0, 0 
df_q = qtable_slice_to_df(RES_T_MAX, demo_bi, demo_ai, demo_bp, demo_ap, demo_turn, MU_D, action_labels)

print("\n================= Q-TABLE SLICE (DEMO) =================")
print(f"State: bid={demo_bi+95}, ask={demo_ai+95}, bp={demo_bp}, ap={demo_ap}, turn={'Alice' if demo_turn==0 else 'Bob'}")
print("Rows are B_Val candidates, columns are actions. Entries are Q(s, a, B_Val).")
print(df_q.to_string(float_format=lambda x: f"{x: .3f}"))
print("========================================================\n")

latex_table = df_q.to_latex(
    float_format=lambda x: f"{x: .3f}",
    escape=False, 
    index=True
)

with open("qtable_demo.tex", "w") as f:
    f.write(latex_table)

print("Wrote LaTeX table to: qtable_demo.tex")
print("\n--- LaTeX table preview ---\n")
print(latex_table)

end_time = time.time()
print(f"### Compilation COMPLETE. Time taken: {end_time - start_time:.2f} seconds ###")
print("==============================================================")

# --- 5. SIMULATION LOOP ---
NOISE_PROB_BOB = 0.10 
actual_mu = [98.0, 102.0] 
bi, ai, bp, ap, turn = 3, 8, 0, 0, 0 
history = []
beliefs = jnp.ones(len(MU_D))/len(MU_D) 

print("\n--- Starting Simulation Loop (12 Turns, 6 Alice turns) ---")

for s in range(12): 
    b_p_101 = float(beliefs[2])
    
    if s % 2 == 0:
        print(f"\n[SIMULATION PROGRESS: Turn {s} / 12 | Bid/Ask: {bi+95}/{ai+95} | P(Bob=101): {b_p_101:.3f}]")
        
    KEY, subkey = jax.random.split(KEY)
    
    if turn == 0:
        q_slice = RES_T_MAX[bi, ai, bp, ap, turn]
        exp_q = jnp.einsum('b, ba -> a', beliefs, q_slice)
        act_optimal = int(jnp.argmax(exp_q))
        act = act_optimal
        
    else:
        mu_actual = actual_mu[turn] 
        rational_a = get_greedy_action(bi, ai, mu_actual)
        random_choice = jax.random.choice(subkey, ACT_D)
        is_random = jax.random.uniform(subkey) < NOISE_PROB_BOB
        
        act_optimal = int(rational_a)
        act = int(jnp.where(is_random, random_choice, rational_a))
    
    pnl = float(get_reward(bi, ai, bp, ap, turn, act, 98.0)) 
    
    if turn == 1:
        likelihoods = jax.vmap(lambda m: get_likelihood(bi, ai, m, act, NOISE_PROB_BOB))(MU_D)
        beliefs = (beliefs * likelihoods)
        beliefs = beliefs / jnp.sum(beliefs) 
        b_p_101 = float(beliefs[2])

    history.append({
        's': s, 
        'b': bi+95, 
        'a': ai+95, 
        'bp': bp, 
        'ap': ap, 
        't': turn, 
        'act': act, 
        'act_opt': act_optimal, 
        'pnl': pnl, 
        'b_p_101': b_p_101,
        'full_beliefs': beliefs.tolist() 
    }) 

    if act == 0 or act == 3: 
        bi, ai, bp, ap = 3, 8, 0, 0 
    else: 
        bi, ai = int(get_next_bi(bi, ai, act)), int(get_next_ai(bi, ai, act))
        if act == 1: bp = turn
        if act == 2: ap = turn
    turn = (turn + 1) % 2 

# --- 6. TABLE PRINTING ---
names = ['Alice', 'Bob']
action_map = {0: 'Sell (0)', 1: '+Bid (1)', 2: '+Ask (2)', 3: 'Buy (3)'}

print("\n## 📊 Simulation Play-by-Play Table (T=6, Gamma=0.95, Final Tradeoff Test)")
print(f"> Alice (True µ=98, Seller) | Bob (True µ=102, Buyer, 10% Noise). Initial B/A: 98/103")
print("-" * 90)
header = "| **Turn** | **Player** | **Bid** | **Ask** | **Action (Code)** | **Optimal?** | **Alice PnL** | **P(Bob=101)** |"
print(header)
print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

for h in history:
    player_name = names[h['t']]
    action_str = action_map[h['act']]
    
    if h['t'] == 0:
        optimal_check = "**Optimal**"
    else:
        optimal_check = "Optimal" if h['act'] == h['act_opt'] else f"**MISTAKE** ({action_map[h['act_opt']]})"
        
    pnl_str = f"{h['pnl']:+.2f}" if h['pnl'] != 0 else "---"
    
    row = (
        f"| {h['s']:<6} | **{player_name}** | {h['b']:<5} | {h['a']:<5} | {action_str:<15} | "
        f"{optimal_check:<12} | {pnl_str:<11} | {h['b_p_101']:.3f} |"
    )
    print(row)

if len(history) < 12:
    print("\nSimulation ended early due to trade.")


# --- 7. DYNAMIC PLOTTING ---

all_turns = np.array([h['s'] for h in history])
bid_prices = np.array([h['b'] for h in history])
ask_prices = np.array([h['a'] for h in history])

trade_data = [h for h in history if h['pnl'] > 0]
trade_turns = np.array([d['s'] for d in trade_data])
trade_prices = np.array([d['a'] for d in trade_data]) 

beliefs_matrix = np.array([h['full_beliefs'] for h in history])
mu_labels = [f'$\\mu={m:.1f}$' for m in MU_D] 

plt.figure(figsize=(10, 6))
sns.heatmap(
    beliefs_matrix,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    cbar_kws={'label': 'Probability $P(\\mu_B)$', 'orientation': 'horizontal'},
    yticklabels=[f'Turn {t}' for t in all_turns],
    xticklabels=mu_labels
)
plt.title('🔥 Heatmap of Alice\'s Belief Distribution Over Bob\'s Value ($\mu_B$)')
plt.ylabel('Turn Number')
plt.xlabel('Hypothesized Bob Value ($\mu_B$)')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(all_turns, ask_prices, marker='^', linestyle='-', color='red', label='Ask Price (Alice\'s Order)')
plt.plot(all_turns, bid_prices, marker='v', linestyle='-', color='blue', label='Bid Price (Bob\'s Order)')
plt.fill_between(all_turns, bid_prices, ask_prices, color='lightgray', alpha=0.5, label='Bid-Ask Spread')

if trade_prices.size > 0:
    first_pnl = trade_data[0]['pnl']
    plt.scatter(trade_turns, trade_prices, color='gold', edgecolor='black', s=150, zorder=5, 
                label=f'Trade Executed (Avg. $+{first_pnl:.2f}$ Profit)')

plt.title('Market Dynamics: Bid/Ask Spread and Trade Execution')
plt.xlabel('Turn Number')
plt.ylabel('Price')
plt.yticks(np.arange(95, 106, 1)) 
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.xticks(all_turns)
plt.show()