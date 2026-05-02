import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# try:
#     CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# except NameError:
#     CURRENT_DIR = os.getcwd()

# PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# from envs import IGTEnv
# ─────────────────────────────────────────────
# IGT Environment
# ─────────────────────────────────────────────
class IGTEnv():
    def __init__(self,
                 mean_reward: np.ndarray,
                 std_reward: np.ndarray,
                 mean_loss: np.ndarray,
                 std_loss: np.ndarray):
        self.num_arms = 4
        self.mean_reward = mean_reward
        self.std_reward = std_reward
        self.mean_loss = mean_loss
        self.std_loss = std_loss
        self.loss_tstamps = {0: np.array([0,1,0,1,0,1,0,1,0,1]),
                             1: np.array([0,0,0,0,1,0,0,0,0,0]),
                             2: np.array([0,1,0,1,0,1,0,1,0,1]),
                             3: np.array([0,0,0,0,1,0,0,0,0,0])}
        for key in self.loss_tstamps:
            np.random.shuffle(self.loss_tstamps[key])
        self.counts = np.zeros((self.num_arms))
        assert self.num_arms == self.mean_reward.shape[0] == self.std_reward.shape[0] == self.mean_loss.shape[0] == self.std_loss.shape[0]
        self.arms = dict(enumerate(zip(self.mean_reward, self.std_reward, self.mean_loss, self.std_loss)))

    def step(self, chosen_arm):
        arm_mean_rew, arm_dev_rew, arm_mean_loss, arm_dev_loss = self.arms[chosen_arm]
        gain = np.random.normal(arm_mean_rew, arm_dev_rew)
        loss = np.random.normal(arm_mean_loss, arm_dev_loss)
        loss = loss * self.loss_tstamps[chosen_arm][int(self.counts[chosen_arm])]
        net  = gain + loss
        self.counts[chosen_arm] += 1
        if self.counts[chosen_arm] == 10:
            self.counts[chosen_arm] = 0
            np.random.shuffle(self.loss_tstamps[chosen_arm])
        return gain, loss, net

    def reset(self):
        self.loss_tstamps = {0: np.array([0,1,0,1,0,1,0,1,0,1]),
                             1: np.array([0,0,0,0,1,0,0,0,0,0]),
                             2: np.array([0,1,0,1,0,1,0,1,0,1]),
                             3: np.array([0,0,0,0,1,0,0,0,0,0])}
        for key in self.loss_tstamps:
            np.random.shuffle(self.loss_tstamps[key])
        self.counts = np.zeros((self.num_arms))
        self.arms = dict(enumerate(zip(self.mean_reward, self.std_reward, self.mean_loss, self.std_loss)))

# ─────────────────────────────────────────────
# Parameters
# A: high reward, frequent large loss  → bad
# B: high reward, rare massive loss    → bad
# C: low reward, frequent small loss   → good
# D: low reward, rare large loss       → good
# ─────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    mean_reward = np.array([100.0, 100.0, 50.0, 50.0]),
    std_reward  = np.array([10.0,  5.0,  10.0,  5.0]),
    mean_loss   = np.array([-250.0, -1250.0, -50.0, -250.0]),
    std_loss    = np.array([10.0,   10.0,   10.0,   10.0]),
)
MAX_TRIALS        = 100
BIN_SIZE          = 20
N_BINS            = MAX_TRIALS // BIN_SIZE
STARTING_BANKROLL = 2000.0
DECK_LABELS       = ["A", "B", "C", "D"]
DECK_COLORS       = ["#E05C5C", "#5C9BE0", "#E0A85C", "#5CB87A"]

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
def init_state():
    if "env" not in st.session_state:
        env = IGTEnv(**DEFAULT_PARAMS)
        st.session_state.env         = env
        st.session_state.bankroll    = STARTING_BANKROLL
        st.session_state.history     = []
        st.session_state.trial       = 0
        st.session_state.last_result = None
        st.session_state.game_over   = False

def reset_game():
    st.session_state.env.reset()
    st.session_state.bankroll    = STARTING_BANKROLL
    st.session_state.history     = []
    st.session_state.trial       = 0
    st.session_state.last_result = None
    st.session_state.game_over   = False

def pick_deck(arm_idx: int):
    if st.session_state.game_over:
        return
    gain, loss, net = st.session_state.env.step(arm_idx)
    st.session_state.bankroll += net
    st.session_state.trial    += 1
    result = dict(
        trial    = st.session_state.trial,
        deck     = DECK_LABELS[arm_idx],
        arm      = arm_idx,
        gain     = gain,
        loss     = loss,
        reward   = net,
        bankroll = st.session_state.bankroll,
    )
    st.session_state.history.append(result)
    st.session_state.last_result = result
    if st.session_state.trial >= MAX_TRIALS:
        st.session_state.game_over = True

# ─────────────────────────────────────────────
# Page config & CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Iowa Gambling Task",
    page_icon="🃏",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f13;
    color: #e8e4dc;
}

.igt-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    color: #7EB8F7;
    line-height: 1.1;
    margin-bottom: 0;
}
.igt-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: #7a7468;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 24px;
}
.bankroll-box {
    background: #1a1a22;
    border: 1px solid #2e2e3a;
    border-radius: 12px;
    padding: 14px 24px;
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 28px;
}
.bankroll-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #5a5860;
    text-transform: uppercase;
}
.bankroll-amount {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.bankroll-trial {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #5a5860;
    margin-left: auto;
    letter-spacing: 0.12em;
}
.result-card {
    background: #1a1a22;
    border-radius: 12px;
    border: 1px solid #2e2e3a;
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.result-deck-badge {
    width: 42px; height: 56px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; font-weight: 900; color: white;
    flex-shrink: 0;
}
.gameover-banner {
    background: linear-gradient(135deg, #1e1e2e, #2a1a2e);
    border: 1px solid #4a2a5a;
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    margin-bottom: 24px;
}
.gameover-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; font-weight: 900;
    color: #c89edc;
    margin-bottom: 8px;
}
.gameover-score {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    color: #e8e4dc;
}
.progress-wrap {
    background: #1a1a22;
    border-radius: 999px;
    height: 6px;
    margin-bottom: 28px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #7EB8F7, #5CB87A);
    transition: width 0.3s ease;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #5a5860;
    text-transform: uppercase;
    margin-bottom: 6px;
    margin-top: 0px;
}
.block-container { max-width: 680px; padding-top: 2rem; }
div[data-testid="stButton"] button {
    background: transparent;
    border: 1px solid #2e2e3a;
    color: #9a9490;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    border-radius: 8px;
    padding: 8px 20px;
    transition: all 0.15s ease;
}
div[data-testid="stButton"] button:hover {
    border-color: #7EB8F7;
    color: #7EB8F7;
    background: rgba(126, 184, 247, 0.07);
}
section[data-testid="stSidebar"] { background: #0d0d10; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────
init_state()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Playfair Display",serif; font-size:1.3rem; font-weight:700; color:#7EB8F7; margin-bottom:8px;'>Instructions</div>
    <div style='font-family:"DM Sans",sans-serif; font-size:0.82rem; color:#9a9490; line-height:1.7;'>
    You start with <b style='color:#e8e4dc'>$2,000</b> in play money.<br><br>
    Each round, pick one of the four decks — <b>A, B, C, or D</b>.<br><br>
    Every pick gives you a <b style='color:#5CB87A'>reward</b>. Some picks also carry a <b style='color:#E05C5C'>penalty</b>.<br><br>
    Your goal is to <b style='color:#e8e4dc'>maximize your bankroll</b> over <b>100 trials</b>.<br><br>
    The decks differ in how often and how severely they penalize you — figure out the pattern.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if st.session_state.history:
        df_side = pd.DataFrame(st.session_state.history)
        counts  = df_side["deck"].value_counts().reindex(DECK_LABELS, fill_value=0)
        st.markdown("<div class='section-label'>Deck selections</div>", unsafe_allow_html=True)
        for i, label in enumerate(DECK_LABELS):
            pct = int(counts[label] / len(df_side) * 100) if len(df_side) > 0 else 0
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                <div style='width:22px; height:22px; background:{DECK_COLORS[i]}; border-radius:4px; display:flex; align-items:center; justify-content:center; font-family:"Playfair Display",serif; font-weight:700; font-size:0.8rem; color:white;'>{label}</div>
                <div style='flex:1; background:#1a1a22; border-radius:999px; height:5px; overflow:hidden;'>
                    <div style='width:{pct}%; background:{DECK_COLORS[i]}; height:100%; border-radius:999px;'></div>
                </div>
                <div style='font-family:"DM Mono",monospace; font-size:0.65rem; color:#5a5860; width:28px; text-align:right;'>{counts[label]}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="igt-title">Iowa Gambling Task</div>', unsafe_allow_html=True)
st.markdown('<div class="igt-subtitle">Decision-making under uncertainty</div>', unsafe_allow_html=True)

# ── Bankroll ──
trial    = st.session_state.trial
bankroll = st.session_state.bankroll
br_color = "#5CB87A" if bankroll >= STARTING_BANKROLL else "#E05C5C"
st.markdown(f"""
<div class="bankroll-box">
    <span class="bankroll-label">Balance</span>
    <span class="bankroll-amount" style="color:{br_color};">${bankroll:,.0f}</span>
    <span class="bankroll-trial">Trial {trial} / {MAX_TRIALS}</span>
</div>
""", unsafe_allow_html=True)

# ── Progress bar ──
pct = int(trial / MAX_TRIALS * 100)
st.markdown(f"""
<div class="progress-wrap">
    <div class="progress-fill" style="width:{pct}%;"></div>
</div>
""", unsafe_allow_html=True)

# ── Last result flash ──
if st.session_state.last_result:
    r          = st.session_state.last_result
    gain       = r["gain"]
    loss       = r["loss"]
    net        = r["reward"]
    deck_color = DECK_COLORS[r["arm"]]
    net_color  = "#5CB87A" if net >= 0 else "#E05C5C"

    gain_str   = f"+${gain:,.0f}"
    loss_str   = f"-${abs(loss):,.0f}" if loss != 0 else "—"
    loss_color = "#E05C5C" if loss != 0 else "#5a5860"
    net_str    = f"+${net:,.0f}" if net >= 0 else f"-${abs(net):,.0f}"

    st.markdown(f"""
    <div class="result-card" style="align-items:center; gap:18px;">
        <div class="result-deck-badge" style="background:{deck_color}; flex-shrink:0;">{r['deck']}</div>
        <div style="flex:1;">
            <div style="font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.18em; color:#5a5860; text-transform:uppercase; margin-bottom:8px;">
                Trial {r['trial']}
            </div>
            <div style="display:flex; gap:24px; align-items:baseline;">
                <div>
                    <div style="font-family:'DM Mono',monospace; font-size:0.58rem; color:#5a5860; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:2px;">Gain</div>
                    <div style="font-family:'DM Mono',monospace; font-size:1.05rem; color:#5CB87A;">{gain_str}</div>
                </div>
                <div style="color:#2e2e3a; font-size:1.2rem;">|</div>
                <div>
                    <div style="font-family:'DM Mono',monospace; font-size:0.58rem; color:#5a5860; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:2px;">Loss</div>
                    <div style="font-family:'DM Mono',monospace; font-size:1.05rem; color:{loss_color};">{loss_str}</div>
                </div>
                <div style="color:#2e2e3a; font-size:1.2rem;">|</div>
                <div>
                    <div style="font-family:'DM Mono',monospace; font-size:0.58rem; color:#5a5860; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:2px;">Net</div>
                    <div style="font-family:'DM Mono',monospace; font-size:1.15rem; font-weight:600; color:{net_color};">{net_str}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Game over banner ──
if st.session_state.game_over:
    final = st.session_state.bankroll
    delta = final - STARTING_BANKROLL
    sign  = "+" if delta >= 0 else ""
    outcome_word = "Well played." if delta >= 0 else "Better luck next time."
    st.markdown(f"""
    <div class="gameover-banner">
        <div class="gameover-title">Game Over</div>
        <div class="gameover-score">Final balance: ${final:,.0f} &nbsp;·&nbsp; {sign}${delta:,.0f} vs start</div>
        <div style='font-family:"DM Sans",sans-serif; font-size:0.82rem; color:#9a9490; margin-top:8px;'>{outcome_word}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Deck buttons ──
if not st.session_state.game_over:
    cols = st.columns(4, gap="small")
    for i, (label, color) in enumerate(zip(DECK_LABELS, DECK_COLORS)):
        with cols[i]:
            st.markdown(f"""
            <style>
            div[data-testid="column"]:nth-child({i+1}) div[data-testid="stButton"] button {{
                background: {color}18;
                border: 2px solid {color}55;
                color: {color};
                font-family: 'Playfair Display', serif;
                font-size: 2rem;
                font-weight: 900;
                width: 100%;
                aspect-ratio: 2/3;
                border-radius: 14px;
                padding: 0;
                letter-spacing: 0;
                transition: all 0.12s ease;
            }}
            div[data-testid="column"]:nth-child({i+1}) div[data-testid="stButton"] button:hover {{
                background: {color}35;
                border-color: {color};
                transform: translateY(-3px);
                box-shadow: 0 10px 28px {color}40;
            }}
            </style>
            """, unsafe_allow_html=True)
            if st.button(label, key=f"deck_{i}", use_container_width=True):
                pick_deck(i)
                st.rerun()

# ── Equity curve (live) ──
if len(st.session_state.history) > 1:
    df = pd.DataFrame(st.session_state.history)

    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#0f0f13")
    ax.set_facecolor("#0f0f13")

    ax.axhline(STARTING_BANKROLL, color="#2e2e3a", linewidth=0.8, linestyle="--")
    ax.fill_between(df["trial"], STARTING_BANKROLL, df["bankroll"],
                    where=(df["bankroll"] >= STARTING_BANKROLL),
                    color="#5CB87A", alpha=0.12)
    ax.fill_between(df["trial"], STARTING_BANKROLL, df["bankroll"],
                    where=(df["bankroll"] < STARTING_BANKROLL),
                    color="#E05C5C", alpha=0.12)

    for j in range(len(df) - 1):
        seg_color = "#5CB87A" if df["bankroll"].iloc[j+1] >= STARTING_BANKROLL else "#E05C5C"
        ax.plot(df["trial"].iloc[j:j+2], df["bankroll"].iloc[j:j+2],
                color=seg_color, linewidth=1.8, solid_capstyle="round")

    for arm_idx, color in enumerate(DECK_COLORS):
        sub = df[df["arm"] == arm_idx]
        ax.scatter(sub["trial"], sub["bankroll"], color=color,
                   s=22, zorder=5, alpha=0.85, linewidths=0)

    ax.set_xlim(1, max(MAX_TRIALS, df["trial"].max()))
    ax.spines[["top","right","left","bottom"]].set_color("#2e2e3a")
    ax.set_xlabel("Trial", color="#5a5860", fontsize=7, labelpad=4)
    ax.set_ylabel("Balance ($)", color="#5a5860", fontsize=7, labelpad=4)
    ax.tick_params(axis='both', colors='#5a5860', labelsize=7)

    patches = [mpatches.Patch(color=DECK_COLORS[i], label=f"Deck {DECK_LABELS[i]}") for i in range(4)]
    ax.legend(handles=patches, loc="upper left", framealpha=0,
              labelcolor="#9a9490", fontsize=6.5, ncol=4,
              handlelength=1, handleheight=0.8)

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Controls ──
st.markdown("<br>", unsafe_allow_html=True)
col_r, _ = st.columns([1, 3])
with col_r:
    if st.button("↺  New Game", use_container_width=True):
        reset_game()
        st.rerun()

# ═══════════════════════════════════════════════════════
# POST-GAME ANALYSIS
# ═══════════════════════════════════════════════════════
if st.session_state.game_over and len(st.session_state.history) == MAX_TRIALS:
    df = pd.DataFrame(st.session_state.history)

    BG         = "#0f0f13"
    bin_labels = [f"{b*BIN_SIZE+1}–{(b+1)*BIN_SIZE}" for b in range(N_BINS)]
    x          = np.arange(N_BINS)

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.spines[["top","right","left","bottom"]].set_color("#2e2e3a")
        ax.tick_params(axis='both', colors='#5a5860', labelsize=7)
        ax.xaxis.label.set_color('#5a5860')
        ax.yaxis.label.set_color('#5a5860')

    # compute bin counts once
    bin_counts = np.zeros((N_BINS, 4), dtype=int)
    for b in range(N_BINS):
        chunk = df[(df["trial"] > b * BIN_SIZE) & (df["trial"] <= (b+1) * BIN_SIZE)]
        for arm_idx in range(4):
            bin_counts[b, arm_idx] = (chunk["arm"] == arm_idx).sum()

    st.divider()
    st.markdown("<div class='section-label' style='margin-top:8px;'>Post-game analysis</div>", unsafe_allow_html=True)

    # ── 1. Deck reveal ──────────────────────────────
    st.markdown("<div class='section-label' style='margin-top:14px;'>Deck reveal</div>", unsafe_allow_html=True)

    reveal_rows = [
        ("A", "Bad",  "High reward, frequent heavy penalties. Net negative over time.",   DECK_COLORS[0]),
        ("B", "Bad",  "High reward, rare but massive penalties. Net negative over time.", DECK_COLORS[1]),
        ("C", "Good", "Low reward, frequent small penalties. Net positive over time.",    DECK_COLORS[2]),
        ("D", "Good", "Low reward, rare large penalties. Net positive over time.",        DECK_COLORS[3]),
    ]
    for deck_label, verdict, desc, color in reveal_rows:
        sub     = df[df["deck"] == deck_label]
        count   = len(sub)
        avg     = sub["reward"].mean() if count > 0 else 0
        v_color = "#5CB87A" if verdict == "Good" else "#E05C5C"
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:14px; padding:12px 16px; background:#1a1a22; border-radius:10px; margin-bottom:8px; border:1px solid #2e2e3a;'>
            <div style='width:36px; height:48px; background:{color}; border-radius:7px; display:flex; align-items:center; justify-content:center; font-family:"Playfair Display",serif; font-weight:900; font-size:1.2rem; color:white; flex-shrink:0;'>{deck_label}</div>
            <div style='flex:1;'>
                <div style='font-family:"DM Sans",sans-serif; font-size:0.8rem; color:#e8e4dc; margin-bottom:2px;'>
                    <span style='color:{v_color}; font-weight:500;'>{verdict} deck</span> — {desc}
                </div>
                <div style='font-family:"DM Mono",monospace; font-size:0.65rem; color:#5a5860;'>
                    Picked {count}× · avg {("+" if avg>=0 else "")}{avg:,.0f} / trial
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 2. Deck selections per bin ──────────────────
    st.markdown("<div class='section-label' style='margin-top:24px;'>Deck selections · per 20-trial bin</div>", unsafe_allow_html=True)

    fig1, ax1 = plt.subplots(figsize=(7, 2.8))
    fig1.patch.set_facecolor(BG)
    w = 0.18
    for arm_idx in range(4):
        offset = (arm_idx - 1.5) * w
        ax1.bar(x + offset, bin_counts[:, arm_idx], width=w,
                color=DECK_COLORS[arm_idx], alpha=0.88,
                label=DECK_LABELS[arm_idx], zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, fontsize=6.5)
    ax1.set_ylabel("# picks", fontsize=7)
    ax1.set_xlabel("Trial bin", fontsize=7)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.axhline(0, color="#2e2e3a", linewidth=0.6)
    ax1.grid(axis='y', color='#2e2e3a', linewidth=0.5, zorder=0)
    ax1.legend(framealpha=0, labelcolor="#9a9490", fontsize=6.5, ncol=4,
               handlelength=1, loc="upper right")
    style_ax(ax1)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig1, use_container_width=True)
    plt.close()

    # ── 3. IGT score per bin ────────────────────────
    st.markdown("""
    <div style='margin-top:18px;'>
        <div class='section-label'>IGT score · per 20-trial bin</div>
        <div style='font-family:"DM Mono",monospace; font-size:0.62rem; color:#5a5860; margin-top:3px; letter-spacing:0.05em;'>
            Score = Advantageous − Disadvantageous &nbsp;=&nbsp; <span style='color:#5CB87A;'>(C+D)</span> − <span style='color:#E05C5C;'>(A+B)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    igt_scores = np.array([
        (bin_counts[b, 2] + bin_counts[b, 3]) - (bin_counts[b, 0] + bin_counts[b, 1])
        for b in range(N_BINS)
    ], dtype=float)
    bar_colors = ["#5CB87A" if s >= 0 else "#E05C5C" for s in igt_scores]

    fig2, ax2 = plt.subplots(figsize=(7, 2.4))
    fig2.patch.set_facecolor(BG)
    bars2 = ax2.bar(x, igt_scores, color=bar_colors, alpha=0.88, zorder=3, width=0.55)
    ax2.axhline(0, color="#7a7468", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels, fontsize=6.5)
    ax2.set_ylabel("IGT score", fontsize=7)
    ax2.set_xlabel("Trial bin", fontsize=7)
    ax2.grid(axis='y', color='#2e2e3a', linewidth=0.5, zorder=0)
    for bar, score in zip(bars2, igt_scores):
        label_y = bar.get_height() + 0.4 if score >= 0 else bar.get_height() - 1.4
        ax2.text(bar.get_x() + bar.get_width() / 2, label_y,
                 f"{int(score):+d}", ha='center', va='bottom',
                 fontsize=6.5, color='#9a9490', fontfamily='monospace')
    style_ax(ax2)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    # ── 4. Event / sequence plot ────────────────────
    st.markdown("<div class='section-label' style='margin-top:18px;'>Trial sequence · red = disadvantageous (A/B) · green = advantageous (C/D)</div>", unsafe_allow_html=True)

    arm_seq   = df["arm"].values
    trial_seq = np.arange(len(arm_seq))
    colors_ev = np.where(np.isin(arm_seq, [0, 1]), '#E05C5C', '#5CB87A')

    fig3, ax3 = plt.subplots(figsize=(7, 2.2))
    fig3.patch.set_facecolor(BG)
    ax3.scatter(trial_seq, arm_seq, c=colors_ev, s=150, marker='|', linewidths=0.8, zorder=3)

    for b in range(1, N_BINS):
        ax3.axvline(b * BIN_SIZE - 0.5, color="#2e2e3a", linewidth=0.6, linestyle="--")
        ax3.text(b * BIN_SIZE - 0.5, 3.65,
                 f"bin {b+1}", ha='center', va='top',
                 fontsize=5.5, color='#5a5860', fontfamily='monospace')

    for y in [0.5, 1.5, 2.5]:
        ax3.axhline(y, color="#1e1e28", linewidth=0.5)

    ax3.set_ylim(-0.6, 4.2)
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['A', 'B', 'C', 'D'], fontsize=7)
    ax3.set_xlabel("Trial", fontsize=7, labelpad=4)
    ax3.set_xlim(-1, MAX_TRIALS)
    ax3.grid(axis='x', color='#1a1a22', linewidth=0.4, zorder=0)

    leg_patches = [
        mpatches.Patch(color='#E05C5C', label='Disadvantageous (A/B)'),
        mpatches.Patch(color='#5CB87A', label='Advantageous (C/D)'),
    ]
    ax3.legend(handles=leg_patches, framealpha=0, labelcolor="#9a9490",
               fontsize=6.5, loc="upper left", handlelength=1)
    style_ax(ax3)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig3, use_container_width=True)
    plt.close()