"""Build the interactive delivery-delay dashboard (single static HTML page).

    python dashboard/build_dashboard.py

Output: dashboard/index.html (hosted on GitHub Pages).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from delay.data import load_orders  # noqa: E402
from delay.features import build_features  # noqa: E402
from delay.model import evaluate, gradient_boosting, time_split  # noqa: E402

INK, SIGNAL, SLATE, LINE, BG = "#0F3D5C", "#E4572E", "#5B6B7A", "#D8DFE6", "#FFFFFF"
FONT = "Archivo, system-ui, sans-serif"


def style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family=FONT, size=13, color="#1F2933"),
        hoverlabel=dict(font_family=FONT, bgcolor="white"),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, ticks="outside", tickcolor=LINE)
    fig.update_yaxes(gridcolor="#EEF1F4", zeroline=False)
    return fig


def html(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displayModeBar": False, "responsive": True})


def main() -> None:
    df = load_orders()
    late_rate = df.is_late.mean()

    # 1. Monthly late rate
    m = (df.set_index("purchased_at").resample("MS")
           .agg(orders=("order_id", "size"), late=("is_late", "mean")))
    m = m[m.orders >= 200]
    f1 = go.Figure()
    f1.add_hline(y=late_rate, line=dict(color=SLATE, dash="dot", width=1))
    f1.add_scatter(x=m.index, y=m.late, mode="lines+markers",
                   line=dict(color=INK, width=2.5), marker=dict(size=6),
                   customdata=m.orders,
                   hovertemplate="%{x|%B %Y}<br>%{y:.1%} en retard<br>%{customdata:,} commandes<extra></extra>")
    peak = m.late.idxmax()
    f1.add_annotation(x=peak, y=m.late.max(), text=f"{m.late.max():.0%}", showarrow=False,
                      yshift=14, font=dict(color=SIGNAL, size=14))
    f1.update_yaxes(tickformat=".0%", rangemode="tozero")
    style(f1)

    # 2. Map by customer state
    geo = json.loads((ROOT / "dashboard" / "brazil_states.geojson").read_text())
    s = df.groupby("customer_state").agg(orders=("order_id", "size"), late=("is_late", "mean"),
                                         days=("actual_days", "median")).reset_index()
    f2 = go.Figure(go.Choropleth(
        geojson=geo, locations=s.customer_state, z=s.late,
        colorscale=[[0, "#EAF0F5"], [0.45, "#7FA3BD"], [0.7, INK], [1, SIGNAL]],
        zmin=0, zmax=0.22, marker_line_color="white", marker_line_width=0.8,
        customdata=np.stack([s.orders, s.days], axis=1),
        hovertemplate="<b>%{location}</b><br>%{z:.1%} en retard<br>"
                      "%{customdata[0]:,} commandes<br>délai médian %{customdata[1]:.0f} j<extra></extra>",
        colorbar=dict(tickformat=".0%", thickness=10, len=0.7, outlinewidth=0),
    ))
    f2.update_geos(fitbounds="locations", visible=False, bgcolor=BG, projection_type="mercator")
    f2.update_layout(geo_domain=dict(x=[0, 0.88], y=[0, 1]))
    style(f2, 380)

    # 3. Review vs delay
    late = df[df.is_late == 1].copy()
    late["bucket"] = pd.cut(late.delay_days, [0, 3, 7, 14, 30, 400],
                            labels=["1-3 j", "4-7 j", "8-14 j", "15-30 j", "30 j +"])
    c = late.groupby("bucket", observed=True).review_score.agg(["mean", "size"])
    x = ["à l'heure"] + list(c.index.astype(str))
    y = [df[df.is_late == 0].review_score.mean()] + list(c["mean"])
    n = [int((df.is_late == 0).sum())] + list(c["size"])
    f3 = go.Figure(go.Bar(x=x, y=y, marker_color=[INK] + [SIGNAL] * len(c), customdata=n,
                          text=[f"{v:.1f}" for v in y], textposition="outside",
                          hovertemplate="%{x}<br>note moyenne %{y:.2f}/5<br>%{customdata:,} commandes<extra></extra>"))
    f3.update_yaxes(range=[0, 5.2], dtick=1)
    style(f3)

    # 4. Where time is lost
    stages = {"approval_days": "validation paiement", "seller_handling_days": "préparation vendeur",
              "carrier_transit_days": "transport"}
    med = df.groupby("is_late")[list(stages)].median()
    f4 = go.Figure()
    for col, color in zip(stages, ["#B8C4CF", INK, SIGNAL]):
        f4.add_bar(y=["À l'heure", "En retard"], x=med[col].values, name=stages[col], orientation="h",
                   marker_color=color, text=[f"{v:.1f} j" if v >= 1 else "" for v in med[col]],
                   textposition="inside", insidetextanchor="middle",
                   hovertemplate=f"{stages[col]} : %{{x:.1f}} jours (médiane)<extra></extra>")
    f4.update_layout(barmode="stack", showlegend=True,
                     legend=dict(orientation="h", y=-0.25, x=0, traceorder="normal"))
    style(f4, 230)
    f4.update_layout(showlegend=True)

    # 5. Model gain curve
    feat = build_features(df)
    sp = time_split(feat)
    model = gradient_boosting().fit(sp.X_train, sp.y_train)
    proba = model.predict_proba(sp.X_test)[:, 1]
    metrics = evaluate(sp.y_test, proba)
    y_sorted = sp.y_test.values[np.argsort(-proba)]
    share = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    capt = np.cumsum(y_sorted) / y_sorted.sum()
    idx = np.linspace(0, len(share) - 1, 200).astype(int)
    f5 = go.Figure()
    f5.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=LINE, dash="dash"),
                   hoverinfo="skip")
    f5.add_scatter(x=share[idx], y=capt[idx], mode="lines", line=dict(color=INK, width=3),
                   hovertemplate="en contrôlant %{x:.0%} des commandes<br>"
                                 "on attrape %{y:.0%} des retards<extra></extra>")
    k = int(0.1 * len(share)) - 1
    f5.add_scatter(x=[share[k]], y=[capt[k]], mode="markers+text", marker=dict(color=SIGNAL, size=11),
                   text=[f"{capt[k]:.0%}"], textposition="middle right", textfont=dict(color=SIGNAL, size=14),
                   hoverinfo="skip")
    f5.update_xaxes(tickformat=".0%", title_text="commandes contrôlées, les plus risquées d'abord")
    f5.update_yaxes(tickformat=".0%", title_text="retards détectés")
    style(f5, 340)

    on_time_score = df[df.is_late == 0].review_score.mean()
    kpis = [
        (f"{len(df):,}".replace(",", " "), "commandes livrées analysées"),
        (f"{late_rate:.1%}".replace(".", ","), "livrées après la date promise"),
        (f"{late.review_score.mean():.2f}".replace(".", ","), f"note moyenne si retard, contre {on_time_score:.2f}".replace(".", ",")),
        (f"x{metrics['lift_top10']:.1f}".replace(".", ","), "de retards détectés vs contrôle au hasard"),
    ]
    tpl = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    out = tpl
    for i, (v, label) in enumerate(kpis, 1):
        out = out.replace(f"{{{{kpi{i}_value}}}}", v).replace(f"{{{{kpi{i}_label}}}}", label)
    for name, fig in {"monthly": f1, "map": f2, "reviews": f3, "stages": f4, "gain": f5}.items():
        out = out.replace(f"{{{{{name}}}}}", html(fig, name))
    out = out.replace("{{auc}}", f"{metrics['roc_auc']:.2f}".replace(".", ","))
    (ROOT / "dashboard" / "index.html").write_text(out, encoding="utf-8")
    print("dashboard/index.html written", {k: round(v, 3) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
