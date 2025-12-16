import jax
import jax.numpy as jnp
from memo import memo

# --- 1. DOMAINS ---
PRICE_D = jnp.arange(21)
TRADER_D = jnp.array([0, 1, 2])
ACT_D = jnp.array([0, 1, 2])
MU_D = jnp.array([96.0, 98.0, 100.0, 102.0, 104.0])

# --- 2. JIT HELPERS ---
@jax.jit
def get_next_bi(bi, ai, a):
    return jnp.where((a == 1) & (bi + 1 < ai - 1), bi + 1, bi)

@jax.jit
def get_next_ai(bi, ai, a):
    return jnp.where((a == 1) & (bi + 1 < ai - 1), ai - 1, ai)

@jax.jit
def get_greedy_action(bi, ai, mu):
    b_val, a_val = bi + 90.0, ai + 90.0
    can_buy = (mu - a_val) > 0
    can_sell = (b_val - mu) > 0
    can_tighten = (bi + 1) < (ai - 1)
    return jnp.where(can_buy, 2, jnp.where(can_sell, 0, jnp.where(can_tighten, 1, 1)))

@jax.jit
def get_reward(bi, ai, turn, a, mu_a):
    b_val, a_val = bi + 90.0, ai + 90.0
    active_r = jnp.where(a == 2, mu_a - a_val, jnp.where(a == 0, b_val - mu_a, 0.0))
    passive_r = jnp.where(a == 2, a_val - mu_a, jnp.where(a == 0, mu_a - b_val, 0.0))
    return jnp.where(turn == 0, active_r, passive_r)

@jax.jit
def update_belief_w(mu_guess, bi, ai, observed_a):
    pred_a = get_greedy_action(bi, ai, mu_guess)
    return jnp.where(pred_a == observed_a, 1.0, 0.01)

@jax.jit
def get_utility(bi, ai, turn, mu_b, mu_c, a_cand, future_q):
    greedy_a = jnp.where(turn == 1, get_greedy_action(bi, ai, mu_b),
                                   get_greedy_action(bi, ai, mu_c))
    is_hero = (turn == 0)
    is_greedy = (a_cand == greedy_a)
    return jnp.where(is_hero | is_greedy, future_q, -1.0e9)

# --- 3. THE MODEL ---
@memo(cache=True)
def Q[bi: PRICE_D, ai: PRICE_D, turn: TRADER_D, mu_b: MU_D, mu_c: MU_D, a: ACT_D](t, alice_mu=100.0):
    alice: knows(bi, ai, turn, mu_b, mu_c, a)
    alice: snapshots_self_as(f)

    # a=1 is "Tighten" (non-terminal). If a is 0 or 2, future value is 0.
    # We use a simple multiplier (a == 1) to mask the future value.

    return alice [
        get_reward(bi, ai, turn, a, alice_mu) + (
            0.0 if t <= 0 else 0.95 * (a == 1) * imagine [
                # 1. State Transitions (Removed {} to avoid NameErrors)
                f: given(bi_ in PRICE_D, wpp=(bi_ == get_next_bi(bi, ai, a))),
                f: given(ai_ in PRICE_D, wpp=(ai_ == get_next_ai(bi, ai, a))),
                f: given(turn_ in TRADER_D, wpp=(turn_ == (turn + 1) % 3)),

                # 2. Choice: Agents maximize utility (Alice) or follow greedy rules (Opponents)
                f: chooses(a_next in ACT_D,
                    to_maximize=get_utility(bi_, ai_, turn_, mu_b, mu_c, a_next,
                                            Q[bi_, ai_, turn_, mu_b, mu_c, a_next](t - 1))
                ),

                # 3. Bayes Update
                f: draws(mu_b_ in MU_D, wpp=update_belief_w(mu_b_, bi_, ai_, a_next)),
                f: draws(mu_c_ in MU_D, wpp=update_belief_w(mu_c_, bi_, ai_, a_next)),

                # 4. Expected Value over next state/action/beliefs
                E[ f [ Q[bi_, ai_, turn_, mu_b_, mu_c_, a_next](t - 1) ] ]
            ]
        )
    ]

# --- 4. EXECUTION ---
print("Compiling and solving Market Model (t=2)...")
results = Q(10).block_until_ready()
print("Success! Table shape:", results.shape)
