import plotly.graph_objects as go
from src.llm import FACTOR_LABELS


def make_shap_chart(top_factors):
    df = top_factors.copy().sort_values('shap_value')
    colors = ['#ff6b6b' if v > 0 else '#55cc88' for v in df['shap_value']]
    labels = [FACTOR_LABELS.get(f, f) for f in df['feature']]

    fig = go.Figure(go.Bar(
        x=df['shap_value'], y=labels, orientation='h',
        marker_color=colors, marker_line_width=0,
        text=[f"{v:+.3f}" for v in df['shap_value']],
        textposition='outside',
        textfont=dict(color='#8b92a5', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#c8cdd8', size=12),
        margin=dict(l=10, r=60, t=10, b=10), height=280,
        xaxis=dict(
            showgrid=True, gridcolor='#2a2f3d', gridwidth=1,
            zeroline=True, zerolinecolor='#4a4f5d', zerolinewidth=1.5,
            tickfont=dict(size=10, color='#8b92a5'),
            title=dict(text='SHAP Value (impact on prediction)', font=dict(size=11, color='#8b92a5'))
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=11))
    )
    return fig


def make_gauge(charge):
    max_val = 65000
    pct     = min(charge / max_val, 1.0)
    color   = '#ff4444' if pct > 0.38 else ('#ff8800' if pct > 0.15 else '#22aa55')

    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=charge,
        number=dict(prefix='$', valueformat=',.0f',
                    font=dict(family='DM Serif Display', size=28, color='#ffffff')),
        gauge=dict(
            axis=dict(range=[0, max_val], tickwidth=1,
                      tickcolor='#2a2f3d', tickfont=dict(color='#8b92a5', size=10), nticks=6),
            bar=dict(color=color, thickness=0.25),
            bgcolor='#161b27', borderwidth=0,
            steps=[
                dict(range=[0, 10000],  color='#0f2d1a'),
                dict(range=[10000, 25000], color='#2a2010'),
                dict(range=[25000, max_val], color='#2d1010'),
            ],
            threshold=dict(line=dict(color='white', width=2), thickness=0.75, value=charge)
        )
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#c8cdd8'),
        margin=dict(l=20, r=20, t=20, b=10), height=200
    )
    return fig