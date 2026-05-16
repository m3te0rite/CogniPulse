import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="N-Arm Bandit", page_icon="🎰", layout="centered")

# ── Session state initialisation ─────────────────────────────────────────────
def init_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized      = True
        st.session_state.true_probs       = np.round(np.random.uniform(0.2, 0.8, 2), 2)
        st.session_state.rewards          = []          # list of (step, reward)
        st.session_state.cumulative       = []          # running total
        st.session_state.choices          = []          # 0 or 1 per step
        st.session_state.arm_pulls        = [0, 0]
        st.session_state.arm_rewards      = [0, 0]
        st.session_state.total_steps      = 20
        st.session_state.game_over        = False

init_state()

# ── Helper ────────────────────────────────────────────────────────────────────
def pull(arm: int):
    if st.session_state.game_over:
        return
    prob   = st.session_state.true_probs[arm]
    reward = int(np.random.random() < prob)
    step   = len(st.session_state.rewards) + 1

    st.session_state.rewards.append(reward)
    st.session_state.choices.append(arm)
    st.session_state.arm_pulls[arm]  += 1
    st.session_state.arm_rewards[arm] += reward

    running_total = (st.session_state.cumulative[-1] if st.session_state.cumulative else 0) + reward
    st.session_state.cumulative.append(running_total)

    if step >= st.session_state.total_steps:
        st.session_state.game_over = True

def reset():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎰 2-Arm Bandit")

# Instructions
with st.expander("📖 How to Play", expanded=True):
    st.markdown(f"""
**Goal:** Maximise your total reward across **{st.session_state.total_steps} rounds**.

**Rules:**
- Each round you pick one of two slot machines (**Arm A** or **Arm B**).
- Each arm pays out **+1** with some hidden probability, or **0** otherwise.
- The probabilities are fixed but unknown — you have to learn them through trial and error.
- The game ends after {st.session_state.total_steps} pulls.

**Strategy tips:**
- *Explore* early: try both arms to estimate which is better.
- *Exploit* later: once you have a good estimate, stick with the better arm.
- This tension between exploration and exploitation is the core of the bandit problem.
    """)

# Progress
steps_taken = len(st.session_state.rewards)
steps_left  = st.session_state.total_steps - steps_taken
st.progress(steps_taken / st.session_state.total_steps,
            text=f"Round {steps_taken} / {st.session_state.total_steps}  |  Rounds left: {steps_left}")

# ── Choice buttons ────────────────────────────────────────────────────────────
if not st.session_state.game_over:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎰 Pull Arm A", use_container_width=True, type="primary"):
            pull(0)
            st.rerun()
    with col2:
        if st.button("🎰 Pull Arm B", use_container_width=True, type="primary"):
            pull(1)
            st.rerun()

    # Live feedback on last pull
    if st.session_state.rewards:
        last_arm    = "A" if st.session_state.choices[-1] == 0 else "B"
        last_reward = st.session_state.rewards[-1]
        if last_reward:
            st.success(f"Round {steps_taken}: Arm {last_arm} → **+1 reward** 🎉")
        else:
            st.warning(f"Round {steps_taken}: Arm {last_arm} → **No reward** 😐")
else:
    st.error("🏁 Game over! See your final results below.")

# ── Live cumulative reward chart ──────────────────────────────────────────────
if st.session_state.cumulative:
    st.subheader("📈 Cumulative Reward Over Time")

    df_live = pd.DataFrame({
        "Round":             range(1, steps_taken + 1),
        "Cumulative Reward": st.session_state.cumulative,
        "Arm":               ["A" if c == 0 else "B" for c in st.session_state.choices],
    })

    fig_live = go.Figure()
    fig_live.add_trace(go.Scatter(
        x=df_live["Round"],
        y=df_live["Cumulative Reward"],
        mode="lines+markers",
        marker=dict(
            color=["#636EFA" if a == "A" else "#EF553B" for a in df_live["Arm"]],
            size=8,
        ),
        line=dict(color="#888", width=2),
        hovertemplate="Round %{x}<br>Total: %{y}<extra></extra>",
    ))

    # Ideal line (always picking the best arm)
    best_prob = max(st.session_state.true_probs)
    ideal_y   = [round(best_prob * r, 2) for r in range(1, steps_taken + 1)]
    fig_live.add_trace(go.Scatter(
        x=list(range(1, steps_taken + 1)),
        y=ideal_y,
        mode="lines",
        line=dict(color="green", dash="dash", width=1.5),
        name="Ideal (greedy)",
        hovertemplate="Round %{x}<br>Ideal: %{y:.2f}<extra></extra>",
    ))

    fig_live.update_layout(
        xaxis_title="Round",
        yaxis_title="Total Reward",
        legend=dict(x=0, y=1),
        margin=dict(t=20, b=40),
        height=300,
    )
    st.plotly_chart(fig_live, use_container_width=True)
    st.caption("🔵 Arm A  🔴 Arm B  — Green dashed line = ideal expected reward if you always pulled the best arm")

# ── Final metrics ─────────────────────────────────────────────────────────────
if st.session_state.game_over:
    st.divider()
    st.subheader("🏆 Final Results")

    total_reward = st.session_state.cumulative[-1]
    max_possible = sum(st.session_state.true_probs[np.argmax(st.session_state.true_probs)]
                       for _ in range(st.session_state.total_steps))
    regret       = round(max_possible - total_reward, 2)
    hit_rate     = round(total_reward / st.session_state.total_steps * 100, 1)

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Reward",     total_reward,       f"/ {st.session_state.total_steps} max")
    m2.metric("Hit Rate",         f"{hit_rate}%",     "rewards / pulls")
    m3.metric("Regret",           regret,             "vs always-best arm", delta_color="inverse")

    st.divider()

    # Reveal the true probabilities
    st.markdown("### 🔍 True Arm Probabilities (revealed)")
    pc1, pc2 = st.columns(2)
    pc1.metric("Arm A true P(reward)", f"{st.session_state.true_probs[0]:.0%}")
    pc2.metric("Arm B true P(reward)", f"{st.session_state.true_probs[1]:.0%}")
    best_arm = "A" if st.session_state.true_probs[0] >= st.session_state.true_probs[1] else "B"
    st.info(f"The **best arm was {best_arm}**. Did you figure it out in time?")

    # Pulls breakdown bar chart
    st.markdown("### 🎯 Arm Pull Distribution")
    df_pulls = pd.DataFrame({
        "Arm":     ["Arm A", "Arm B"],
        "Pulls":   st.session_state.arm_pulls,
        "Rewards": st.session_state.arm_rewards,
    })
    df_pulls["Misses"] = df_pulls["Pulls"] - df_pulls["Rewards"]

    fig_pulls = go.Figure(data=[
        go.Bar(name="Rewards", x=df_pulls["Arm"], y=df_pulls["Rewards"],
               marker_color=["#636EFA", "#EF553B"]),
        go.Bar(name="Misses",  x=df_pulls["Arm"], y=df_pulls["Misses"],
               marker_color=["#A0AEC0", "#CBD5E0"]),
    ])
    fig_pulls.update_layout(
        barmode="stack",
        yaxis_title="Count",
        margin=dict(t=20, b=40),
        height=300,
        legend=dict(x=1, y=1),
    )
    st.plotly_chart(fig_pulls, use_container_width=True)

    # Estimated vs true probability
    st.markdown("### 📊 Your Estimated Probabilities vs Truth")
    est_a = (st.session_state.arm_rewards[0] / st.session_state.arm_pulls[0]
             if st.session_state.arm_pulls[0] else 0)
    est_b = (st.session_state.arm_rewards[1] / st.session_state.arm_pulls[1]
             if st.session_state.arm_pulls[1] else 0)

    df_est = pd.DataFrame({
        "Arm":  ["Arm A", "Arm A", "Arm B", "Arm B"],
        "Type": ["True", "Estimated", "True", "Estimated"],
        "P":    [st.session_state.true_probs[0], est_a,
                 st.session_state.true_probs[1], est_b],
    })
    fig_est = px.bar(df_est, x="Arm", y="P", color="Type", barmode="group",
                     color_discrete_map={"True": "#2ECC71", "Estimated": "#F39C12"},
                     labels={"P": "Probability"})
    fig_est.update_layout(margin=dict(t=20, b=40), height=300, yaxis_range=[0, 1])
    st.plotly_chart(fig_est, use_container_width=True)

    # Regret over time
    st.markdown("### 📉 Cumulative Regret Over Time")
    best_p   = max(st.session_state.true_probs)
    regrets  = []
    running  = 0
    for arm in st.session_state.choices:
        running += best_p - st.session_state.true_probs[arm]
        regrets.append(round(running, 4))

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(
        x=list(range(1, steps_taken + 1)),
        y=regrets,
        fill="tozeroy",
        line=dict(color="#E74C3C", width=2),
        hovertemplate="Round %{x}<br>Regret: %{y:.3f}<extra></extra>",
    ))
    fig_reg.update_layout(
        xaxis_title="Round",
        yaxis_title="Cumulative Regret",
        margin=dict(t=20, b=40),
        height=280,
    )
    st.plotly_chart(fig_reg, use_container_width=True)
    st.caption("Regret = gap between your actual rewards and what you'd have earned always picking the best arm.")

    st.divider()
    if st.button("🔄 Play Again", type="primary", use_container_width=True):
        reset()
        st.rerun()