from functools import cache
from memo import memo
import jax
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

hidden_mu_vals = np.array([92.0, 105.0, 107.0])
belief_mu_vals = np.array([95.0, 105.0, 107.0, 110.0, 106.0])
trader_names = ["A", "B", "C"]
num_turns = 3
TRUE_MU = 100.0

starting_bid = 90.0
starting_ask = 110.0
num_prices = int(starting_ask - starting_bid + 1)
num_mu_states = len(belief_mu_vals)

# split up state space into smaller domains for speed.
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


def visualize_simulation(history_data):
    """Create comprehensive visualizations of the simulation."""

    # Extract data
    all_steps = []
    all_rounds = []
    all_prob_b = []
    all_prob_c = []
    all_bids = []
    all_asks = []
    all_actions = []
    all_traders = []
    all_q_expected = []

    for round_data in history_data:
        for step_data in round_data['steps']:
            all_steps.append(step_data['step'])
            all_rounds.append(round_data['round_num'])
            all_prob_b.append(onp.array(step_data['prob_b']))
            all_prob_c.append(onp.array(step_data['prob_c']))
            all_bids.append(step_data['bid'])
            all_asks.append(step_data['ask'])
            all_actions.append(step_data['action'])
            all_traders.append(step_data['trader'])
            if 'q_expected' in step_data:
                all_q_expected.append(step_data['q_expected'])

    # Convert to numpy arrays
    prob_b_matrix = onp.array(all_prob_b)
    prob_c_matrix = onp.array(all_prob_c)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Belief Heatmap for Agent B
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(prob_b_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest', origin='lower')
    ax1.set_xlabel('Step (across all rounds)')
    ax1.set_ylabel('Belief μ Value')
    ax1.set_title('Belief Distribution for Agent B (prob_b)', fontsize=12, fontweight='bold')
    ax1.set_yticks(range(len(belief_mu_vals)))
    ax1.set_yticklabels([f'{mu:.1f}' for mu in belief_mu_vals])
    plt.colorbar(im1, ax=ax1, label='Probability')

    # Add vertical lines for round boundaries
    step_idx = 0
    for round_data in history_data:
        step_idx += len(round_data['steps'])
        if step_idx < len(all_steps):
            ax1.axvline(step_idx - 0.5, color='blue', linestyle='--', alpha=0.5, linewidth=1)

    # 2. Belief Heatmap for Agent C
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(prob_c_matrix.T, aspect='auto', cmap='YlGnBu', interpolation='nearest', origin='lower')
    ax2.set_xlabel('Step (across all rounds)')
    ax2.set_ylabel('Belief μ Value')
    ax2.set_title('Belief Distribution for Agent C (prob_c)', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(len(belief_mu_vals)))
    ax2.set_yticklabels([f'{mu:.1f}' for mu in belief_mu_vals])
    plt.colorbar(im2, ax=ax2, label='Probability')

    # Add vertical lines for round boundaries
    step_idx = 0
    for round_data in history_data:
        step_idx += len(round_data['steps'])
        if step_idx < len(all_steps):
            ax2.axvline(step_idx - 0.5, color='blue', linestyle='--', alpha=0.5, linewidth=1)

    # 3. Most Likely Belief Over Time
    ax3 = fig.add_subplot(gs[0, 2])
    most_likely_b = [belief_mu_vals[onp.argmax(pb)] for pb in all_prob_b]
    most_likely_c = [belief_mu_vals[onp.argmax(pc)] for pc in all_prob_c]
    true_b = float(hidden_mu_vals[1])
    true_c = float(hidden_mu_vals[2])

    ax3.plot(most_likely_b, 'o-', label='Most Likely μ for B', color='orange', markersize=4)
    ax3.plot(most_likely_c, 's-', label='Most Likely μ for C', color='cyan', markersize=4)
    ax3.axhline(true_b, color='orange', linestyle='--', alpha=0.5, label=f'True μ_B = {true_b}')
    ax3.axhline(true_c, color='cyan', linestyle='--', alpha=0.5, label=f'True μ_C = {true_c}')
    ax3.set_xlabel('Step (across all rounds)')
    ax3.set_ylabel('μ Value')
    ax3.set_title('Most Likely Belief vs True Values', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Spread Evolution
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(all_bids, 'o-', label='Bid', color='green', markersize=3, linewidth=1.5)
    ax4.plot(all_asks, 's-', label='Ask', color='red', markersize=3, linewidth=1.5)
    ax4.fill_between(range(len(all_bids)), all_bids, all_asks, alpha=0.2, color='gray', label='Spread')
    ax4.set_xlabel('Step (across all rounds)')
    ax4.set_ylabel('Price')
    ax4.set_title('Bid-Ask Spread Evolution', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add round boundaries
    step_idx = 0
    for i, round_data in enumerate(history_data):
        step_idx += len(round_data['steps'])
        if step_idx < len(all_steps):
            ax4.axvline(step_idx - 0.5, color='blue', linestyle='--', alpha=0.3, linewidth=1)
            ax4.text(step_idx - len(round_data['steps'])/2, ax4.get_ylim()[1] * 0.95,
                    f'Round {round_data["round_num"]}', ha='center', fontsize=8)

    # 5. Action Sequence
    ax5 = fig.add_subplot(gs[2, 0])
    action_colors = {'Tighten': 'blue', 'BUY': 'green', 'SELL': 'red'}
    action_map = {0: 'Tighten', 1: 'BUY', 2: 'SELL'}
    action_names = [action_map.get(a, 'Unknown') for a in all_actions]
    colors = [action_colors.get(name, 'gray') for name in action_names]

    ax5.scatter(range(len(all_actions)), all_actions, c=colors, s=50, alpha=0.6)
    ax5.set_xlabel('Step (across all rounds)')
    ax5.set_ylabel('Action')
    ax5.set_yticks([0, 1, 2])
    ax5.set_yticklabels(['Tighten', 'BUY', 'SELL'])
    ax5.set_title('Action Sequence', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add legend
    patches = [mpatches.Patch(color=color, label=name) for name, color in action_colors.items()]
    ax5.legend(handles=patches, loc='upper right')

    # 6. Trader Turn Sequence
    ax6 = fig.add_subplot(gs[2, 1])
    trader_map = {'A': 0, 'B': 1, 'C': 2}
    trader_nums = [trader_map.get(t, 0) for t in all_traders]
    trader_colors_map = {'A': 'purple', 'B': 'orange', 'C': 'cyan'}
    trader_colors = [trader_colors_map.get(t, 'gray') for t in all_traders]

    ax6.scatter(range(len(trader_nums)), trader_nums, c=trader_colors, s=50, alpha=0.6)
    ax6.set_xlabel('Step (across all rounds)')
    ax6.set_ylabel('Trader')
    ax6.set_yticks([0, 1, 2])
    ax6.set_yticklabels(['A', 'B', 'C'])
    ax6.set_title('Trader Turn Sequence', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')

    # 7. Belief Entropy (Uncertainty)
    ax7 = fig.add_subplot(gs[2, 2])
    def entropy(probs):
        probs = onp.array(probs)
        probs = probs[probs > 0]  # Avoid log(0)
        return -onp.sum(probs * onp.log(probs))

    entropy_b = [entropy(pb) for pb in all_prob_b]
    entropy_c = [entropy(pc) for pc in all_prob_c]

    ax7.plot(entropy_b, 'o-', label='Entropy(B)', color='orange', markersize=3)
    ax7.plot(entropy_c, 's-', label='Entropy(C)', color='cyan', markersize=3)
    ax7.set_xlabel('Step (across all rounds)')
    ax7.set_ylabel('Entropy (bits)')
    ax7.set_title('Belief Uncertainty (Entropy)', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Q-Value Expectations (if available)
    if all_q_expected:
        ax8 = fig.add_subplot(gs[3, :])
        q_expected_array = onp.array(all_q_expected)
        ax8.plot(q_expected_array[:, 0], 'o-', label='Q(Tighten)', markersize=3)
        ax8.plot(q_expected_array[:, 1], 's-', label='Q(BUY)', markersize=3)
        ax8.plot(q_expected_array[:, 2], '^-', label='Q(SELL)', markersize=3)
        ax8.set_xlabel('Step (Hero turns only)')
        ax8.set_ylabel('Expected Q-Value')
        ax8.set_title('Expected Q-Values for Hero Actions', fontsize=12, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

    plt.suptitle('POMDP Simulation Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('pomdp_simulation_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to 'pomdp_simulation_analysis.png'")
    plt.show()

def run_simulation(Q_vals, visualize=True):
    total_true_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
    total_hidden_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}

    prob_b = np.ones(num_mu_states) / num_mu_states
    prob_c = np.ones(num_mu_states) / num_mu_states

    # History tracking for visualization
    history_data = []

    print(f"\n{'='*20} STARTING SIMULATION (3 ROUNDS) {'='*20}")
    curr_turn = 0

    for round_num in range(1, 4):
        round_steps = []  # Track steps in this round
        curr_bid = starting_bid
        curr_ask = starting_ask

        no_progress_count = 0

        print(f"\n--- ROUND {round_num} START: Bid {curr_bid} | Ask {curr_ask} ---")

        for step in range(30):
            trader = trader_names[curr_turn]
            prev_trader_idx = (curr_turn - 1) % 3
            prev_trader = trader_names[prev_trader_idx]

            b_idx = int(curr_bid - starting_bid)
            a_idx = int(curr_ask - starting_bid)

            if curr_turn == 0:

                q_s = Q_vals[b_idx, a_idx, curr_turn]

                q_weighted_c = np.tensordot(prob_c, q_s, axes=([0], [1]))
                q_expected = np.tensordot(prob_b, q_weighted_c, axes=([0], [0]))

                if (curr_bid + 1) >= (curr_ask - 1):
                    q_expected[0] = -1e9

                action = int(np.argmax(q_expected))

                top_b = belief_mu_vals[np.argmax(prob_b)]
                top_c = belief_mu_vals[np.argmax(prob_c)]
                source = f"POMDP (Belief Top B: {top_b}, Top C: {top_c})"

                # Track Q-values for visualization
                q_expected_track = onp.array(q_expected)

            else:

                mu_i = hidden_mu_vals[curr_turn]
                action = int(get_greedy_action_for_mu(b_idx, a_idx, mu_i))
                source = f"Greedy (Mu={mu_i})"

                # UPDATE PERSISTENT BELIEFS
                lik_fn = lambda idx: update_belief_w(idx, b_idx, a_idx, action)
                if curr_turn == 1:
                    likelihoods = np.array([lik_fn(i) for i in range(num_mu_states)])
                    prob_b = (prob_b * likelihoods) / np.sum(prob_b * likelihoods)
                elif curr_turn == 2:
                    likelihoods = np.array([lik_fn(i) for i in range(num_mu_states)])
                    prob_c = (prob_c * likelihoods) / np.sum(prob_c * likelihoods)

            act_str = ["Tighten", "BUY", "SELL"][action]
            print(f"Step {step} | {trader} [{source}] -> {act_str}")

            # Record step data for visualization
            step_data = {
                'step': step,
                'round_num': round_num,
                'trader': trader,
                'action': action,
                'bid': curr_bid,
                'ask': curr_ask,
                'prob_b': onp.array(prob_b),
                'prob_c': onp.array(prob_c),
            }
            if curr_turn == 0 and 'q_expected_track' in locals():
                step_data['q_expected'] = q_expected_track
            round_steps.append(step_data)

            if action == 1: # BUY
                r_actor = TRUE_MU - curr_ask
                r_counter = curr_ask - TRUE_MU
                r_actor_hidden = float(hidden_mu_vals[curr_turn]) - curr_ask
                r_counter_hidden = curr_ask - float(hidden_mu_vals[prev_trader_idx])

                total_true_rewards[trader] += r_actor
                total_true_rewards[prev_trader] += r_counter
                total_hidden_rewards[trader] += r_actor_hidden
                total_hidden_rewards[prev_trader] += r_counter_hidden

                print(f" >>> EXECUTION: {trader} BUYS from {prev_trader} @ {curr_ask}")
                print(f"     Round Profits: {trader}={r_actor}, {prev_trader}={r_counter}")
                curr_turn = (curr_turn + 1) % 3
                history_data.append({'round_num': round_num, 'steps': round_steps})
                break # ROUND ENDS

            elif action == 2: # SELL
                r_actor = curr_bid - TRUE_MU
                r_counter = TRUE_MU - curr_bid
                r_actor_hidden = curr_bid - float(hidden_mu_vals[curr_turn])
                r_counter_hidden = float(hidden_mu_vals[prev_trader_idx]) - curr_bid

                total_true_rewards[trader] += r_actor
                total_true_rewards[prev_trader] += r_counter
                total_hidden_rewards[trader] += r_actor_hidden
                total_hidden_rewards[prev_trader] += r_counter_hidden

                print(f" >>> EXECUTION: {trader} SELLS to {prev_trader} @ {curr_bid}")
                print(f"     Round Profits: {trader}={r_actor}, {prev_trader}={r_counter}")
                curr_turn = (curr_turn + 1) % 3
                history_data.append({'round_num': round_num, 'steps': round_steps})
                break # ROUND ENDS

            elif action == 0: # TIGHTEN
                if (curr_bid + 1) < (curr_ask - 1):
                    curr_bid += 1; curr_ask -= 1
                    print(f"     Update: Spread tightens to {curr_bid} ... {curr_ask}")
                else:
                    no_progress_count += 1
                    if no_progress_count >= 3:
                         print(f"     Market Locked. Round Void.")
                         history_data.append({'round_num': round_num, 'steps': round_steps})
                         break

            curr_turn = (curr_turn + 1) % 3

        # If round ended without break (timeout), record it
        if len(history_data) < round_num:
            history_data.append({'round_num': round_num, 'steps': round_steps})

    print("\n" + "="*60)
    print("FINAL CUMULATIVE SCORES:", total_true_rewards)
    print("HIDDEN SCORES:", total_hidden_rewards)
    print("="*60)

    # Generate visualizations
    if visualize and history_data:
        visualize_simulation(history_data)

    return history_data, total_true_rewards, total_hidden_rewards


if __name__ == "__main__":
    print("Compiling Policy...")
    # This will now compile MUCH faster
    Q_vals = Q(3).block_until_ready()
    print("Policy compiled. Simulating...")
    run_simulation(Q_vals)
