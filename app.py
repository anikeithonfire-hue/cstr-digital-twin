"""
CSTR Digital Twin — Live Process Dashboard
===========================================
Run this file:   python app.py
Then open:       http://127.0.0.1:8050

Controls
────────
  Setpoint slider     : change the target reactor temperature
  Inject flow drop    : simulates 50 % loss of feed flow (common pump fault)
  Inject cooling fault: simulates partial cooling system failure
  Clear fault         : returns the process to normal operating conditions
"""

from collections import deque

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots

from fault_detector import FaultDetector
from kalman import KalmanFilter1D
from pid import PIDController
from simulator import CSTRSimulator

# ══════════════════════════════════════════════════════════════
#  Global process objects  (shared across Dash callbacks)
# ══════════════════════════════════════════════════════════════
MAX_PTS = 300          # number of data points kept in memory per channel

sim      = CSTRSimulator()
kf_T     = KalmanFilter1D(Q=1e-4,  R=0.64,    x0=350.0)
kf_Ca    = KalmanFilter1D(Q=1e-6,  R=6.4e-5,  x0=0.5)
pid      = PIDController(setpoint=350.0)
detector = FaultDetector()

# Rolling data buffers
buf = {k: deque(maxlen=MAX_PTS) for k in [
    "time", "T_meas", "T_filt", "Ca_meas", "Ca_filt",
    "P_meas", "F_meas", "Tc", "setpoint", "fault_score",
]}

Tc_now          = 300.0   # current coolant temperature
last_flow_btn   = 0
last_heat_btn   = 0
last_clear_btn  = 0

# ══════════════════════════════════════════════════════════════
#  Layout helpers
# ══════════════════════════════════════════════════════════════
CARD_STYLE = {
    "background" : "#f4f4f4",
    "borderRadius": "8px",
    "padding"    : "12px 16px",
}

BTN_STYLE = {
    "fontSize"    : "13px",
    "marginRight" : "8px",
    "cursor"      : "pointer",
}


def metric_card(label, value, unit, danger=False):
    color = "#c0392b" if danger else "#111111"
    return html.Div([
        html.Div(label, style={"fontSize": "12px", "color": "#888", "marginBottom": "4px"}),
        html.Div(
            f"{value} {unit}",
            style={"fontSize": "22px", "fontWeight": "500", "color": color},
        ),
    ], style=CARD_STYLE)


# ══════════════════════════════════════════════════════════════
#  App layout
# ══════════════════════════════════════════════════════════════
app = dash.Dash(__name__, title="CSTR Digital Twin")

app.layout = html.Div([

    # ── Header ───────────────────────────────────────────────
    html.Div([
        html.H2(
            "CSTR Digital Twin — Live Process Monitor",
            style={"margin": "0", "fontWeight": "500", "fontSize": "20px"},
        ),
        html.P(
            "Continuous Stirred Tank Reactor  ·  Kalman Filter  ·  PID Control  ·  Fault Detection",
            style={"margin": "4px 0 0", "fontSize": "13px", "color": "#888"},
        ),
    ], style={"padding": "20px 24px 16px", "borderBottom": "0.5px solid #e5e5e5"}),

    # ── Metric cards ─────────────────────────────────────────
    html.Div(
        id="metric-cards",
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(4, 1fr)",
            "gap": "12px",
            "padding": "16px 24px",
        },
    ),

    # ── Status bar ───────────────────────────────────────────
    html.Div(id="status-bar", style={"padding": "0 24px 12px"}),

    # ── Main chart ───────────────────────────────────────────
    dcc.Graph(
        id="main-chart",
        config={"displayModeBar": False},
        style={"padding": "0 12px"},
    ),

    # ── Controls ─────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label(
                "Temperature setpoint (K)",
                style={"fontSize": "13px", "color": "#666", "display": "block", "marginBottom": "6px"},
            ),
            dcc.Slider(
                id="setpoint-slider",
                min=330, max=380, step=1, value=350,
                marks={330: "330 K", 350: "350 K", 365: "365 K", 380: "380 K"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], style={"padding": "0 24px", "marginBottom": "16px"}),

        html.Div([
            html.Button("Inject flow drop",    id="btn-flow",  n_clicks=0, style=BTN_STYLE),
            html.Button("Inject cooling fault", id="btn-heat",  n_clicks=0, style=BTN_STYLE),
            html.Button("Clear fault",          id="btn-clear", n_clicks=0, style=BTN_STYLE),
        ], style={"padding": "0 24px 20px"}),
    ]),

    # ── Interval timer (fires every 200 ms) ──────────────────
    dcc.Interval(id="interval", interval=200, n_intervals=0),

], style={"fontFamily": "system-ui, -apple-system, sans-serif", "background": "#fafafa", "minHeight": "100vh"})


# ══════════════════════════════════════════════════════════════
#  Main callback — runs every 200 ms
# ══════════════════════════════════════════════════════════════
@app.callback(
    Output("metric-cards", "children"),
    Output("status-bar",   "children"),
    Output("main-chart",   "figure"),
    Input("interval",        "n_intervals"),
    Input("setpoint-slider", "value"),
    Input("btn-flow",        "n_clicks"),
    Input("btn-heat",        "n_clicks"),
    Input("btn-clear",       "n_clicks"),
)
def update(n_intervals, setpoint, n_flow, n_heat, n_clear):
    global Tc_now, last_flow_btn, last_heat_btn, last_clear_btn

    # ── Handle fault buttons ──────────────────────────────────
    if n_flow  > last_flow_btn:
        sim.inject_fault("flow_drop")
        last_flow_btn = n_flow

    if n_heat  > last_heat_btn:
        sim.inject_fault("heat_loss")
        last_heat_btn = n_heat

    if n_clear > last_clear_btn:
        sim.clear_fault()
        last_clear_btn = n_clear

    # ── Update PID setpoint ───────────────────────────────────
    pid.set_setpoint(setpoint)

    # ── Advance simulation (3 steps per UI tick) ─────────────
    STEPS = 3
    for _ in range(STEPS):
        current_T_estimate = kf_T.x if buf["T_filt"] else 350.0

        Tc_now, _err = pid.compute(current_T_estimate, dt=sim.dt)
        meas = sim.step(Tc_now)

        T_f  = kf_T.update(meas["T_meas"])
        Ca_f = kf_Ca.update(meas["Ca_meas"])
        detector.update(T_f, Ca_f, meas["P_meas"], meas["F_meas"])

        buf["time"].append(meas["time"])
        buf["T_meas"].append(meas["T_meas"])
        buf["T_filt"].append(T_f)
        buf["Ca_meas"].append(meas["Ca_meas"])
        buf["Ca_filt"].append(Ca_f)
        buf["P_meas"].append(meas["P_meas"])
        buf["F_meas"].append(meas["F_meas"])
        buf["Tc"].append(Tc_now)
        buf["setpoint"].append(float(setpoint))
        buf["fault_score"].append(detector.score)

    # ── Convenience lists for plotting ───────────────────────
    t        = list(buf["time"])
    t_filt   = list(buf["T_filt"])
    ca_filt  = list(buf["Ca_filt"])
    p_meas   = list(buf["P_meas"])
    f_meas   = list(buf["F_meas"])
    tc_list  = list(buf["Tc"])
    sp_list  = list(buf["setpoint"])
    fscores  = list(buf["fault_score"])

    latest_T  = round(t_filt[-1],  2) if t_filt  else 350.0
    latest_Ca = round(ca_filt[-1], 4) if ca_filt else 0.5
    latest_P  = round(p_meas[-1],  1) if p_meas  else 101.3
    latest_F  = round(f_meas[-1],  1) if f_meas  else 100.0
    fault_now = detector.fault_active

    # ── Metric cards ─────────────────────────────────────────
    cards = [
        metric_card("Temperature (filtered)", f"{latest_T:.1f}",  "K",      danger=fault_now),
        metric_card("Concentration Ca",        f"{latest_Ca:.4f}", "mol/L"),
        metric_card("Pressure",                f"{latest_P:.1f}",  "kPa"),
        metric_card("Flow rate",               f"{latest_F:.1f}",  "L/min"),
    ]

    # ── Status bar ───────────────────────────────────────────
    dot_color  = "#c0392b" if fault_now else "#27ae60"
    bar_bg     = "#fff7f7" if fault_now else "#f7fff9"
    bar_border = "#e74c3c" if fault_now else "#2ecc71"
    status_bar = html.Div([
        html.Span("● ", style={"color": dot_color,   "fontSize": "16px"}),
        html.Span(detector.status,
                  style={"fontSize": "14px", "fontWeight": "500", "color": dot_color}),
        html.Span(
            f"  |  Coolant Tc: {round(Tc_now, 1)} K  |  Time: {round(sim.t, 1)} min",
            style={"fontSize": "13px", "color": "#888", "marginLeft": "12px"},
        ),
    ], style={
        "background"  : bar_bg,
        "border"      : f"0.5px solid {bar_border}",
        "borderRadius": "8px",
        "padding"     : "10px 16px",
    })

    # ── Build 3×2 subplot figure ─────────────────────────────
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Temperature (K)",           "Concentration Ca (mol/L)",
            "PID coolant output Tc (K)", "Pressure (kPa)",
            "Flow rate (L/min)",         "Fault z-score",
        ],
        vertical_spacing=0.13,
        horizontal_spacing=0.09,
    )

    # Row 1 — Temperature
    fig.add_trace(go.Scatter(
        x=t, y=list(buf["T_meas"]),
        mode="lines", name="T raw",
        line=dict(color="rgba(52,152,219,0.30)", width=1),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=t_filt,
        mode="lines", name="T filtered (Kalman)",
        line=dict(color="#2980b9", width=2),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t, y=sp_list,
        mode="lines", name="Setpoint",
        line=dict(color="#e74c3c", width=1.5, dash="dash"),
        showlegend=False,
    ), row=1, col=1)

    # Row 1 — Concentration
    fig.add_trace(go.Scatter(
        x=t, y=list(buf["Ca_meas"]),
        mode="lines",
        line=dict(color="rgba(39,174,96,0.30)", width=1),
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=t, y=ca_filt,
        mode="lines",
        line=dict(color="#27ae60", width=2),
        showlegend=False,
    ), row=1, col=2)

    # Row 2 — PID coolant output
    fig.add_trace(go.Scatter(
        x=t, y=tc_list,
        mode="lines",
        line=dict(color="#8e44ad", width=2),
        showlegend=False,
    ), row=2, col=1)

    # Row 2 — Pressure
    fig.add_trace(go.Scatter(
        x=t, y=p_meas,
        mode="lines",
        line=dict(color="#e67e22", width=1.5),
        showlegend=False,
    ), row=2, col=2)

    # Row 3 — Flow rate
    fig.add_trace(go.Scatter(
        x=t, y=f_meas,
        mode="lines",
        line=dict(color="#16a085", width=1.5),
        showlegend=False,
    ), row=3, col=1)

    # Row 3 — Fault z-score
    fig.add_trace(go.Scatter(
        x=t, y=fscores,
        mode="lines",
        line=dict(color="#e74c3c", width=1.5),
        showlegend=False,
    ), row=3, col=2)
    if t:
        fig.add_hline(
            y=3.5,
            line_dash="dash", line_color="#e74c3c", line_width=1,
            row=3, col=2,
        )

    # Fault highlight band on all subplots
    if fault_now and len(t) >= 2:
        fault_start = t[max(0, len(t) - 20)]
        for row in range(1, 4):
            for col in range(1, 3):
                fig.add_vrect(
                    x0=fault_start, x1=t[-1],
                    fillcolor="rgba(231,76,60,0.06)",
                    line_width=0,
                    row=row, col=col,
                )

    fig.update_layout(
        height=580,
        margin=dict(l=40, r=20, t=40, b=20),
        paper_bgcolor="#fafafa",
        plot_bgcolor="#ffffff",
        font=dict(family="system-ui, sans-serif", size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", linecolor="#e0e0e0", showline=True)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", linecolor="#e0e0e0", showline=True)

    return cards, status_bar, fig


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  CSTR Digital Twin  —  starting server")
    print("  Open your browser at:  http://127.0.0.1:8050")
    print("=" * 50 + "\n")
    app.run(debug=False)

    