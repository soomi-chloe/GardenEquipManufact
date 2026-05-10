import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import copy
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="총괄생산계획 시각화 웹앱",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CSS 스타일
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size:1.8rem; font-weight:700; color:#1E3A5F; margin-bottom:0.2rem;}
    .sub-header  {font-size:0.95rem; color:#555; margin-bottom:1.5rem;}
    .kpi-card    {background:#F0F4FA; border-radius:10px; padding:1rem 1.2rem; text-align:center;}
    .kpi-label   {font-size:0.8rem; color:#666; margin-bottom:0.3rem;}
    .kpi-value   {font-size:1.6rem; font-weight:700; color:#1E3A5F;}
    .kpi-unit    {font-size:0.75rem; color:#888;}
    .warn-box    {background:#FFF3CD; border-left:4px solid #FFC107;
                  padding:0.7rem 1rem; border-radius:4px; margin:0.4rem 0;}
    .danger-box  {background:#FFE0E0; border-left:4px solid #DC3545;
                  padding:0.7rem 1rem; border-radius:4px; margin:0.4rem 0;}
    .ok-box      {background:#D4EDDA; border-left:4px solid #28A745;
                  padding:0.7rem 1rem; border-radius:4px; margin:0.4rem 0;}
    .section-title {font-size:1.1rem; font-weight:600; color:#1E3A5F;
                    border-bottom:2px solid #1E3A5F; padding-bottom:0.3rem; margin:1rem 0 0.8rem 0;}
</style>
""", unsafe_allow_html=True)

SCENARIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios.json")

# ─────────────────────────────────────────
# 시나리오 저장/로드
# ─────────────────────────────────────────
def load_scenarios():
    if os.path.exists(SCENARIO_FILE):
        with open(SCENARIO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_scenarios(scenarios):
    with open(SCENARIO_FILE, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────
# 총괄생산계획 계산 엔진 (Pyomo LP/IP)
# ─────────────────────────────────────────
def run_app_plan(params: dict, solver_type: str = "LP") -> dict:
    """
    params 키:
      demand        : list[int]   # 기간별 수요
      init_workers  : int         # 초기 종업원 수
      init_inv      : int         # 초기 재고
      min_final_inv : int         # 최종 목표 재고
      init_shortage : int         # 초기 부족재고 (보통 0)
      reg_wage      : float       # 정규 임금 (천원/시간)
      ot_wage       : float       # 초과근무 임금 (천원/시간)
      hire_cost     : float       # 고용 비용 (천원/인)
      fire_cost     : float       # 해고 비용 (천원/인)
      inv_cost      : float       # 재고유지비용 (천원/개/월)
      back_cost     : float       # 부재고비용 (천원/개/월)
      mat_cost      : float       # 재료비 (천원/개)
      sub_cost      : float       # 하청비용 (천원/개)  ← 하청 추가비용 (생산비 초과분)
      work_days     : int         # 작업일수 (일/월)
      work_hours    : int         # 작업시간 (시간/일)
      max_ot        : float       # 초과시간 상한 (시간/인/월)
      std_time      : float       # 작업표준시간 (시간/개)
    """
    try:
        from pyomo.environ import (
            ConcreteModel, Var, Objective, Constraint, SolverFactory,
            NonNegativeReals, NonNegativeIntegers, minimize, value, sum_product
        )
    except ImportError:
        return {"error": "Pyomo가 설치되어 있지 않습니다."}

    D = params["demand"]
    TH = len(D)
    T = range(1, TH + 1)
    TIME = range(0, TH + 1)

    type_var = NonNegativeIntegers if solver_type == "IP" else NonNegativeReals

    # 비용 계수
    reg_wage   = params["reg_wage"]
    ot_wage    = params["ot_wage"]
    hire_cost  = params["hire_cost"]
    fire_cost  = params["fire_cost"]
    inv_cost   = params["inv_cost"]
    back_cost  = params["back_cost"]
    mat_cost   = params["mat_cost"]
    sub_cost   = params["sub_cost"]
    work_days  = params["work_days"]
    work_hours = params["work_hours"]
    max_ot     = params["max_ot"]
    std_time   = params["std_time"]

    # 정규시간 단위당 노동비 = 임금 × 작업시간 × 작업일수 × std_time (월 기준 1인당)
    # 목적함수 계수 (천원)
    # W_t: 1인당 월 정규임금 = reg_wage × work_hours × work_days
    w_coeff  = reg_wage * work_hours * work_days   # 종업원 1인당 월 정규 노동비
    o_coeff  = ot_wage                              # 초과시간 1시간당
    h_coeff  = hire_cost
    l_coeff  = fire_cost
    i_coeff  = inv_cost
    s_coeff  = back_cost
    p_coeff  = mat_cost
    c_coeff  = sub_cost

    m = ConcreteModel()
    m.W = Var(TIME, domain=type_var, bounds=(0, None))
    m.H = Var(TIME, domain=type_var, bounds=(0, None))
    m.L = Var(TIME, domain=type_var, bounds=(0, None))
    m.P = Var(TIME, domain=type_var, bounds=(0, None))
    m.I = Var(TIME, domain=type_var, bounds=(0, None))
    m.S = Var(TIME, domain=type_var, bounds=(0, None))
    m.C = Var(TIME, domain=type_var, bounds=(0, None))
    m.O = Var(TIME, domain=type_var, bounds=(0, None))

    # 목적함수: 총비용 최소화
    m.Cost = Objective(
        expr=sum(
            w_coeff * m.W[t] + o_coeff * m.O[t] + h_coeff * m.H[t] +
            l_coeff * m.L[t] + i_coeff * m.I[t] + s_coeff * m.S[t] +
            p_coeff * m.P[t] + c_coeff * m.C[t]
            for t in T
        ),
        sense=minimize
    )

    # 제약조건
    # 1. 노동력 균형: W_t = W_{t-1} + H_t - L_t
    m.labor = Constraint(T, rule=lambda m, t: m.W[t] == m.W[t-1] + m.H[t] - m.L[t])

    # 2. 생산능력: P_t <= (1/std_time)*work_hours*work_days*W_t + O_t/std_time
    cap_reg = work_hours * work_days / std_time
    m.capacity = Constraint(T, rule=lambda m, t:
        m.P[t] <= cap_reg * m.W[t] + m.O[t] / std_time)

    # 3. 재고균형: I_t = I_{t-1} + P_t + C_t - D[t-1] - S_{t-1} + S_t
    m.inventory = Constraint(T, rule=lambda m, t:
        m.I[t] == m.I[t-1] + m.P[t] + m.C[t] - D[t-1] - m.S[t-1] + m.S[t])

    # 4. 초과근무 상한: O_t <= max_ot * W_t
    m.overtime = Constraint(T, rule=lambda m, t: m.O[t] <= max_ot * m.W[t])

    # 초기값
    m.W_0 = Constraint(rule=m.W[0] == params["init_workers"])
    m.I_0 = Constraint(rule=m.I[0] == params["init_inv"])
    m.S_0 = Constraint(rule=m.S[0] == params["init_shortage"])

    # 최종 제약
    m.last_inv      = Constraint(rule=m.I[TH] >= params["min_final_inv"])
    m.last_shortage = Constraint(rule=m.S[TH] == 0)

    # 비음수
    m.H[0].fix(0); m.L[0].fix(0)
    m.P[0].fix(0); m.C[0].fix(0); m.O[0].fix(0)

    solver = SolverFactory("glpk")
    result = solver.solve(m, tee=False)

    status = str(result.solver.termination_condition)
    if status != "optimal":
        return {"error": f"풀이 실패: {status}"}

    total_cost = value(m.Cost)

    workers  = [value(m.W[t]) for t in TIME]
    hire     = [value(m.H[t]) for t in TIME]
    fire     = [value(m.L[t]) for t in TIME]
    prod     = [value(m.P[t]) for t in TIME]
    inv      = [value(m.I[t]) for t in TIME]
    shortage = [value(m.S[t]) for t in TIME]
    outsrc   = [value(m.C[t]) for t in TIME]
    overtime = [value(m.O[t]) for t in TIME]

    # 기간별 비용 분해
    periods = list(T)
    cost_breakdown = []
    for t in T:
        cost_breakdown.append({
            "정규노동비":  round(w_coeff * workers[t], 1),
            "초과근무비":  round(o_coeff * overtime[t], 1),
            "고용비":      round(h_coeff * hire[t], 1),
            "해고비":      round(l_coeff * fire[t], 1),
            "재고유지비":  round(i_coeff * inv[t], 1),
            "부재고비":    round(s_coeff * shortage[t], 1),
            "재료비":      round(p_coeff * prod[t], 1),
            "하청비":      round(c_coeff * outsrc[t], 1),
        })

    return {
        "status": "optimal",
        "total_cost": round(total_cost, 1),
        "workers":    [round(v, 2) for v in workers],
        "hire":       [round(v, 2) for v in hire],
        "fire":       [round(v, 2) for v in fire],
        "prod":       [round(v, 2) for v in prod],
        "inv":        [round(v, 2) for v in inv],
        "shortage":   [round(v, 2) for v in shortage],
        "outsrc":     [round(v, 2) for v in outsrc],
        "overtime":   [round(v, 2) for v in overtime],
        "cost_breakdown": cost_breakdown,
        "demand":     [0] + D,
    }

# ─────────────────────────────────────────
# 결과 테이블 생성
# ─────────────────────────────────────────
def make_result_df(res: dict, period_unit: str) -> pd.DataFrame:
    TH = len(res["demand"]) - 1
    labels = [f"{period_unit}{t}" for t in range(1, TH+1)]
    rows = []
    for i, t in enumerate(range(1, TH+1)):
        rows.append({
            "기간":      labels[i],
            "수요":      res["demand"][t],
            "생산량":    res["prod"][t],
            "재고":      res["inv"][t],
            "부족재고":  res["shortage"][t],
            "종업원":    res["workers"][t],
            "고용":      res["hire"][t],
            "해고":      res["fire"][t],
            "외주":      res["outsrc"][t],
            "초과시간":  res["overtime"][t],
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────
def plot_demand_prod_inv(res, period_labels):
    TH = len(res["demand"]) - 1
    t_idx = list(range(1, TH+1))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        subplot_titles=["수요 · 생산 · 재고 추이", "부족재고(결품)"],
        vertical_spacing=0.12
    )

    # 수요
    fig.add_trace(go.Scatter(
        x=period_labels, y=[res["demand"][t] for t in t_idx],
        name="수요", line=dict(color="#E74C3C", width=2.5, dash="dash"),
        mode="lines+markers"
    ), row=1, col=1)

    # 생산
    fig.add_trace(go.Bar(
        x=period_labels, y=[res["prod"][t] for t in t_idx],
        name="생산량", marker_color="#3498DB", opacity=0.7
    ), row=1, col=1)

    # 재고 (선)
    fig.add_trace(go.Scatter(
        x=period_labels, y=[res["inv"][t] for t in t_idx],
        name="기말재고", line=dict(color="#2ECC71", width=2),
        mode="lines+markers", yaxis="y2"
    ), row=1, col=1)

    # 결품
    shortage_vals = [res["shortage"][t] for t in t_idx]
    colors = ["#DC3545" if v > 0 else "#ADB5BD" for v in shortage_vals]
    fig.add_trace(go.Bar(
        x=period_labels, y=shortage_vals,
        name="부족재고", marker_color=colors
    ), row=2, col=1)

    fig.update_layout(
        height=500, hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=60, b=20)
    )
    return fig

def plot_cost_breakdown(cost_breakdown, period_labels):
    cost_cols = ["정규노동비","초과근무비","고용비","해고비","재고유지비","부재고비","재료비","하청비"]
    colors    = ["#3498DB","#F39C12","#2ECC71","#E74C3C","#9B59B6","#E67E22","#1ABC9C","#95A5A6"]

    fig = go.Figure()
    for col, color in zip(cost_cols, colors):
        # cost_breakdown은 딕셔너리 리스트 — 키 없으면 0으로 처리
        y_vals = [row.get(col, 0) for row in cost_breakdown]
        fig.add_trace(go.Bar(name=col, x=period_labels, y=y_vals, marker_color=color))

    fig.update_layout(
        barmode="stack", height=360,
        title="기간별 비용 구성",
        xaxis_title="기간", yaxis_title="비용 (천원)",
        legend=dict(orientation="h", y=-0.3),
        margin=dict(t=50, b=80)
    )
    return fig

def plot_cost_pie(cost_breakdown):
    df = pd.DataFrame(cost_breakdown)
    cost_cols = ["정규노동비","초과근무비","고용비","해고비","재고유지비","부재고비","재료비","하청비"]
    totals = {c: df[c].sum() for c in cost_cols if df[c].sum() > 0}
    fig = px.pie(
        names=list(totals.keys()), values=list(totals.values()),
        title="비용 구성 비율", hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=340, margin=dict(t=50, b=20))
    return fig

def plot_workers(res, period_labels):
    TH = len(res["demand"]) - 1
    t_idx = list(range(1, TH+1))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=period_labels, y=[res["workers"][t] for t in t_idx],
        name="종업원 수", marker_color="#3498DB"
    ))
    fig.add_trace(go.Bar(
        x=period_labels, y=[res["hire"][t] for t in t_idx],
        name="고용", marker_color="#2ECC71"
    ))
    fig.add_trace(go.Bar(
        x=period_labels, y=[-res["fire"][t] for t in t_idx],
        name="해고", marker_color="#E74C3C"
    ))
    fig.update_layout(
        barmode="overlay", height=320,
        title="인력 변동 추이",
        yaxis_title="인원 (명)", hovermode="x unified",
        legend=dict(orientation="h"),
        margin=dict(t=50, b=20)
    )
    return fig

# ─────────────────────────────────────────
# 시나리오 비교
# ─────────────────────────────────────────
def compare_scenarios(scenarios, selected_names):
    rows = []
    for name in selected_names:
        s = scenarios[name]
        res = s.get("result")
        if not res or "error" in res:
            continue
        TH = len(res["demand"]) - 1
        total_shortage = sum(res["shortage"][1:])
        avg_inv = np.mean(res["inv"][1:])
        max_inv = max(res["inv"][1:])
        rows.append({
            "시나리오":    name,
            "총비용(천원)": res["total_cost"],
            "총부족재고":   total_shortage,
            "평균재고":     round(avg_inv, 1),
            "최대재고":     max_inv,
            "계획기간":     TH,
            "풀이방식":     s.get("solver_type", "-"),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────
# 이상 구간 탐지
# ─────────────────────────────────────────
def detect_anomalies(res, inv_upper, period_labels):
    alerts = []
    TH = len(res["demand"]) - 1
    for i, t in enumerate(range(1, TH+1)):
        if res["shortage"][t] > 0:
            alerts.append(("danger", f"⚠️ {period_labels[i]}: 부족재고 {res['shortage'][t]:.0f}개 발생"))
        if inv_upper and res["inv"][t] > inv_upper:
            alerts.append(("warn", f"📦 {period_labels[i]}: 재고 {res['inv'][t]:.0f}개 — 상한({inv_upper}) 초과"))
    return alerts

# ═══════════════════════════════════════════
# 메인 UI
# ═══════════════════════════════════════════
st.markdown('<div class="main-header">🏭 총괄생산계획 시각화 웹앱</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">원예장비 제조업체 — Pyomo 기반 LP/IP 최적화</div>', unsafe_allow_html=True)

if "scenarios" not in st.session_state:
    st.session_state.scenarios = load_scenarios()
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_params" not in st.session_state:
    st.session_state.last_params = {}

# ─────────────────────────────────────────
# 사이드바: 입력
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 계획 입력")

    with st.expander("📅 기간/단위 설정", expanded=True):
        period_unit = st.selectbox("타임버킷", ["월", "주"])
        num_periods = st.slider("계획 기간 수", 2, 24, 6)

    with st.expander("📊 수요 데이터 입력", expanded=True):
        st.caption("기간별 수요를 입력하세요.")

        # 기본 수요 예시 (6개월)
        default_demand = [1600, 3000, 3200, 3800, 2200, 2200]

        demand_vals = []
        cols_per_row = 3
        period_labels_input = [f"{period_unit}{t}" for t in range(1, num_periods+1)]
        rows_needed = (num_periods + cols_per_row - 1) // cols_per_row
        for r in range(rows_needed):
            cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                idx = r * cols_per_row + c
                if idx < num_periods:
                    default_val = default_demand[idx] if idx < len(default_demand) else 2000
                    val = cols[c].number_input(
                        period_labels_input[idx], min_value=0,
                        value=default_val, step=100, key=f"d_{idx}"
                    )
                    demand_vals.append(val)

        if demand_vals:
            st.caption(f"합계: **{sum(demand_vals):,}** | 평균: **{np.mean(demand_vals):,.0f}**")

    with st.expander("🏭 생산/인력 파라미터", expanded=False):
        st.markdown("**초기 상태**")
        init_workers = st.number_input("초기 종업원 수 (명)", 1, 500, 80)
        init_inv     = st.number_input("초기 재고 (개)", 0, 100000, 1000, step=100)
        min_final_inv = st.number_input("최종 목표 재고 (개)", 0, 100000, 500, step=100)

        st.markdown("**생산 능력**")
        work_days  = st.number_input("작업일수 (일/월)", 1, 31, 20)
        work_hours = st.number_input("작업시간 (시간/일)", 1, 24, 8)
        std_time   = st.number_input("작업표준시간 (시간/개)", 0.1, 20.0, 4.0, step=0.5)
        max_ot     = st.number_input("초과시간 상한 (시간/인/월)", 0, 200, 10)

    with st.expander("💰 비용 파라미터", expanded=False):
        use_template = st.checkbox("강의록 기본값 템플릿 적용", value=True)
        if use_template:
            reg_wage  = 4.0;  ot_wage  = 6.0
            hire_cost = 300.0; fire_cost = 500.0
            inv_cost  = 2.0;  back_cost = 5.0
            mat_cost  = 10.0; sub_cost  = 30.0
            st.info("강의록 예제 기본값 적용됨")
        reg_wage  = st.number_input("정규임금 (천원/시간)", 0.0, 100.0, reg_wage if use_template else 4.0, step=0.5)
        ot_wage   = st.number_input("초과근무임금 (천원/시간)", 0.0, 100.0, ot_wage if use_template else 6.0, step=0.5)
        hire_cost = st.number_input("고용비용 (천원/인)", 0.0, 5000.0, hire_cost if use_template else 300.0, step=50.0)
        fire_cost = st.number_input("해고비용 (천원/인)", 0.0, 5000.0, fire_cost if use_template else 500.0, step=50.0)
        inv_cost  = st.number_input("재고유지비용 (천원/개/월)", 0.0, 100.0, inv_cost if use_template else 2.0, step=0.5)
        back_cost = st.number_input("부재고비용 (천원/개/월)", 0.0, 100.0, back_cost if use_template else 5.0, step=0.5)
        mat_cost  = st.number_input("재료비 (천원/개)", 0.0, 500.0, mat_cost if use_template else 10.0, step=1.0)
        sub_cost  = st.number_input("하청 추가비용 (천원/개)", 0.0, 500.0, sub_cost if use_template else 30.0, step=1.0)

    with st.expander("🔧 계산 설정", expanded=False):
        solver_type = st.selectbox("풀이 방식", ["LP (실수 최적화)", "IP (정수 최적화)"])
        solver_key  = "LP" if "LP" in solver_type else "IP"
        inv_upper   = st.number_input("재고 과다 기준 (0=미사용)", 0, 999999, 0, step=500)
        inv_upper   = inv_upper if inv_upper > 0 else None

    run_btn = st.button("🚀 계획 수립 실행", type="primary", use_container_width=True)

# ─────────────────────────────────────────
# 계산 실행
# ─────────────────────────────────────────
params = {
    "demand": demand_vals,
    "init_workers": init_workers, "init_inv": init_inv,
    "min_final_inv": min_final_inv, "init_shortage": 0,
    "reg_wage": reg_wage, "ot_wage": ot_wage,
    "hire_cost": hire_cost, "fire_cost": fire_cost,
    "inv_cost": inv_cost, "back_cost": back_cost,
    "mat_cost": mat_cost, "sub_cost": sub_cost,
    "work_days": work_days, "work_hours": work_hours,
    "max_ot": max_ot, "std_time": std_time,
}

period_labels = [f"{period_unit}{t}" for t in range(1, num_periods+1)]

if run_btn:
    # 유효성 검증
    errors = []
    if any(v < 0 for v in demand_vals):
        errors.append("수요에 음수값이 있습니다.")
    if init_inv < 0:
        errors.append("초기 재고는 0 이상이어야 합니다.")
    if errors:
        for e in errors:
            st.error(e)
    else:
        with st.spinner("Pyomo로 최적화 중..."):
            result = run_app_plan(params, solver_key)
        st.session_state.last_result = result
        st.session_state.last_params = copy.deepcopy(params)
        st.session_state.last_solver  = solver_key

# ─────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 대시보드", "📋 결과 테이블", "💾 시나리오 관리", "⚖️ 시나리오 비교"])

# ───── TAB 1: 대시보드 ─────
with tab1:
    res = st.session_state.last_result

    if res is None:
        st.info("왼쪽에서 수요와 파라미터를 입력한 후 **계획 수립 실행** 버튼을 눌러주세요.")
    elif "error" in res:
        st.error(f"계산 오류: {res['error']}")
    else:
        TH = len(res["demand"]) - 1

        # KPI 카드
        total_shortage = sum(res["shortage"][1:])
        avg_inv = np.mean(res["inv"][1:])
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">총 비용</div>
            <div class="kpi-value">{res['total_cost']:,.0f}</div>
            <div class="kpi-unit">천원</div></div>""", unsafe_allow_html=True)
        with k2:
            color = "#DC3545" if total_shortage > 0 else "#2ECC71"
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">총 부족재고</div>
            <div class="kpi-value" style="color:{color}">{total_shortage:,.0f}</div>
            <div class="kpi-unit">개</div></div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">평균 재고</div>
            <div class="kpi-value">{avg_inv:,.0f}</div>
            <div class="kpi-unit">개</div></div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="kpi-card"><div class="kpi-label">계획 기간</div>
            <div class="kpi-value">{TH}</div>
            <div class="kpi-unit">{period_unit}</div></div>""", unsafe_allow_html=True)

        st.markdown("")

        # 이상 구간 알림
        alerts = detect_anomalies(res, inv_upper, period_labels)
        if alerts:
            st.markdown('<div class="section-title">🔔 이상 구간 알림</div>', unsafe_allow_html=True)
            for level, msg in alerts:
                box_class = "danger-box" if level == "danger" else "warn-box"
                st.markdown(f'<div class="{box_class}">{msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-box">✅ 이상 구간 없음 — 결품 0건, 재고 기준 이상 없음</div>', unsafe_allow_html=True)

        st.markdown("")

        # 수요-생산-재고 차트
        st.markdown('<div class="section-title">📉 수요 · 생산 · 재고 추이</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_demand_prod_inv(res, period_labels), use_container_width=True)

        # 비용 구성
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown('<div class="section-title">💸 기간별 비용 구성</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_cost_breakdown(res["cost_breakdown"], period_labels), use_container_width=True)
        with col_r:
            st.markdown('<div class="section-title">🥧 비용 비율</div>', unsafe_allow_html=True)
            st.plotly_chart(plot_cost_pie(res["cost_breakdown"]), use_container_width=True)

        # 인력 차트
        st.markdown('<div class="section-title">👷 인력 변동 추이</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_workers(res, period_labels), use_container_width=True)

# ───── TAB 2: 결과 테이블 ─────
with tab2:
    res = st.session_state.last_result
    if res is None:
        st.info("계획을 먼저 수립해주세요.")
    elif "error" in res:
        st.error(res["error"])
    else:
        st.markdown('<div class="section-title">📋 기간별 결과 테이블</div>', unsafe_allow_html=True)
        df_res = make_result_df(res, period_unit)

        # 결품 행 강조
        def highlight_shortage(row):
            if row["부족재고"] > 0:
                return ["background-color: #FFE0E0"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_res.style.apply(highlight_shortage, axis=1).format({
                col: "{:,.1f}" for col in df_res.select_dtypes("number").columns
            }),
            use_container_width=True, height=400
        )

        # 비용 상세 테이블
        st.markdown('<div class="section-title">💰 기간별 비용 상세</div>', unsafe_allow_html=True)
        df_cost = pd.DataFrame(res["cost_breakdown"])
        df_cost = df_cost.drop(columns=["기간"], errors="ignore")
        df_cost.insert(0, "기간", period_labels)
        total_row = {"기간": "합계"}
        for c in df_cost.columns[1:]:
            total_row[c] = df_cost[c].sum()
        df_cost = pd.concat([df_cost, pd.DataFrame([total_row])], ignore_index=True)

        st.dataframe(df_cost.set_index("기간").style.format("{:,.1f}"), use_container_width=True)

        # 다운로드
        csv_data = df_res.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "⬇️ 결과 테이블 CSV 다운로드",
            data=csv_data.encode("utf-8-sig"),
            file_name=f"app_result_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# ───── TAB 3: 시나리오 관리 ─────
with tab3:
    st.markdown('<div class="section-title">💾 시나리오 저장 / 불러오기</div>', unsafe_allow_html=True)

    res = st.session_state.last_result
    col_save, col_load = st.columns(2)

    with col_save:
        st.subheader("현재 계획 저장")
        if res and "error" not in res:
            sc_name = st.text_input("시나리오 이름", value=f"시나리오_{datetime.now().strftime('%m%d_%H%M')}")
            sc_memo = st.text_area("메모 (선택)", height=80)
            if st.button("💾 저장", use_container_width=True):
                sc_data = {
                    "name": sc_name,
                    "memo": sc_memo,
                    "saved_at": datetime.now().isoformat(),
                    "params": copy.deepcopy(st.session_state.last_params),
                    "result": copy.deepcopy(res),
                    "solver_type": st.session_state.get("last_solver", "LP"),
                    "period_unit": period_unit,
                }
                st.session_state.scenarios[sc_name] = sc_data
                save_scenarios(st.session_state.scenarios)
                st.success(f"'{sc_name}' 저장 완료!")
        else:
            st.info("계획 수립 후 저장할 수 있습니다.")

    with col_load:
        st.subheader("저장된 시나리오 목록")
        scenarios = st.session_state.scenarios
        if not scenarios:
            st.info("저장된 시나리오가 없습니다.")
        else:
            for name, sc in list(scenarios.items()):
                with st.container():
                    c1, c2, c3 = st.columns([4, 2, 2])
                    c1.write(f"**{name}**")
                    c1.caption(sc.get("memo", "") or sc.get("saved_at", "")[:16])
                    if c2.button("불러오기", key=f"load_{name}"):
                        st.session_state.last_result = sc["result"]
                        st.session_state.last_params  = sc["params"]
                        st.session_state.last_solver   = sc.get("solver_type", "LP")
                        st.success(f"'{name}' 불러오기 완료! 대시보드 탭에서 확인하세요.")
                    if c3.button("삭제", key=f"del_{name}"):
                        del st.session_state.scenarios[name]
                        save_scenarios(st.session_state.scenarios)
                        st.rerun()
                    st.divider()

    # 복제 & 재계산
    st.markdown('<div class="section-title">🔁 시나리오 복제 및 재계산</div>', unsafe_allow_html=True)
    if scenarios:
        clone_src = st.selectbox("복제할 시나리오 선택", list(scenarios.keys()))
        clone_name = st.text_input("새 시나리오 이름", value=f"{clone_src}_복제")
        if st.button("📋 복제 후 현재 파라미터로 재계산", use_container_width=True):
            new_params = copy.deepcopy(scenarios[clone_src]["params"])
            # 현재 sidebar 수요만 교체
            new_params["demand"] = demand_vals
            with st.spinner("재계산 중..."):
                new_res = run_app_plan(new_params, solver_key)
            sc_data = {
                "name": clone_name,
                "memo": f"{clone_src} 복제 후 재계산",
                "saved_at": datetime.now().isoformat(),
                "params": new_params,
                "result": new_res,
                "solver_type": solver_key,
                "period_unit": period_unit,
            }
            st.session_state.scenarios[clone_name] = sc_data
            save_scenarios(st.session_state.scenarios)
            if "error" not in new_res:
                st.success(f"'{clone_name}' 복제 완료! 총비용: {new_res['total_cost']:,.0f} 천원")
            else:
                st.error(new_res["error"])
    else:
        st.info("저장된 시나리오가 있어야 복제할 수 있습니다.")

# ───── TAB 4: 시나리오 비교 ─────
with tab4:
    st.markdown('<div class="section-title">⚖️ 시나리오 비교</div>', unsafe_allow_html=True)
    scenarios = st.session_state.scenarios
    if len(scenarios) < 2:
        st.info("비교하려면 시나리오가 2개 이상 필요합니다.")
    else:
        selected = st.multiselect("비교할 시나리오 선택 (2개 이상)", list(scenarios.keys()),
                                   default=list(scenarios.keys())[:min(3, len(scenarios))])
        if len(selected) >= 2:
            df_cmp = compare_scenarios(scenarios, selected)
            st.dataframe(df_cmp.set_index("시나리오").style.format({
                "총비용(천원)": "{:,.0f}",
                "총부족재고": "{:,.0f}",
                "평균재고": "{:,.1f}",
                "최대재고": "{:,.0f}",
            }), use_container_width=True)

            # 총비용 비교 차트
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                x=df_cmp["시나리오"], y=df_cmp["총비용(천원)"],
                marker_color=px.colors.qualitative.Set2[:len(selected)],
                text=df_cmp["총비용(천원)"].apply(lambda x: f"{x:,.0f}"),
                textposition="outside"
            ))
            fig_cmp.update_layout(
                title="시나리오별 총비용 비교", yaxis_title="총비용 (천원)",
                height=360, margin=dict(t=50, b=20)
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

            # 재고 추이 비교
            st.markdown("**재고 추이 비교**")
            fig_inv = go.Figure()
            for name in selected:
                sc = scenarios[name]
                res = sc.get("result")
                if not res or "error" in res:
                    continue
                TH = len(res["demand"]) - 1
                pu = sc.get("period_unit", "월")
                xlabels = [f"{pu}{t}" for t in range(1, TH+1)]
                fig_inv.add_trace(go.Scatter(
                    x=xlabels, y=res["inv"][1:], name=name, mode="lines+markers"
                ))
            fig_inv.update_layout(
                title="시나리오별 재고 추이",
                yaxis_title="기말재고 (개)", height=360,
                legend=dict(orientation="h"), margin=dict(t=50, b=20)
            )
            st.plotly_chart(fig_inv, use_container_width=True)

            # 결품 비교
            st.markdown("**결품(부족재고) 비교**")
            fig_sh = go.Figure()
            for name in selected:
                sc = scenarios[name]
                res = sc.get("result")
                if not res or "error" in res:
                    continue
                TH = len(res["demand"]) - 1
                pu = sc.get("period_unit", "월")
                xlabels = [f"{pu}{t}" for t in range(1, TH+1)]
                fig_sh.add_trace(go.Bar(
                    x=xlabels, y=res["shortage"][1:], name=name
                ))
            fig_sh.update_layout(
                barmode="group", title="시나리오별 부족재고 비교",
                yaxis_title="부족재고 (개)", height=340,
                legend=dict(orientation="h"), margin=dict(t=50, b=20)
            )
            st.plotly_chart(fig_sh, use_container_width=True)
        else:
            st.info("2개 이상 시나리오를 선택해주세요.")