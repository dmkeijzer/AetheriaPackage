import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy import linspace

def PlotContour(xp, y, z, xtit, ytit, title, x):
    fig = go.Figure(data=go.Heatmap(
        x=xp,
        y=y,
        z=z,
        connectgaps=False,
        colorscale='Picnic',
        zsmooth='best'
    ), layout=go.Layout(
        xaxis=dict(title=xtit, autorange='reversed'),
        yaxis=dict(title=ytit),
        title=f'Cross Sectional {title} Stress Distribution at {x = } m'
    ))
    return fig

def NormalStressPlot(oxx, x, Ch, Ca, shape, tsk, **others):
    zi = list(linspace(Ch - Ca, Ch, 1200, dtype='float'))
    yi = list(linspace(-Ch, Ch, 600, dtype='float'))
    o_zy = [[None if abs(y_i) > shape(z_i) else oxx(z_i, y_i)(x)*1e-6 for z_i in zi] for y_i in yi]
    return PlotContour(zi, yi, o_zy,
        'Chordwise Distribution [m]',
        'Vertical Distribution [m]',
        'Normal', x)

def ShearStressPlot(oxx, x, Ch, Ca, shape, tsk, tsp, **others):
    zi = list(linspace(Ch - Ca, Ch, 1800, dtype='float'))
    yi = list(linspace(-Ch, Ch, 1200, dtype='float'))
    o_zy = [[None if abs(y_i) > shape(z_i) or (abs(y_i) < shape(z_i) - tsk and not -tsp < z_i < tsp) \
        else oxx(z_i, y_i)(x)*1e-6 for z_i in zi] for y_i in yi]
    return PlotContour(zi, yi, o_zy,
        'Chordwise Distribution [m]',
        'Vertical Distribution [m]',
        'Shear', x)

def InternalLoading(x0, x1, **Loads):
    titles = [k + f' [kN{"" if k[0].upper() == "V" else "m"}]' for k in Loads]
    fig = make_subplots(rows=len(list(Loads.keys())), cols=1, shared_xaxes=True, vertical_spacing=0.02)
    xs = linspace(x0, x1, 5000)
    for i, (li, load) in enumerate(zip(Loads.keys(), Loads.values())):
        fig.append_trace(go.Scatter(
            x=xs,
            y=[load(xi)*1e-3 for xi in xs],
            name=titles[i],
        ), row=i+1, col=1)
        fig.update_yaxes(row=i+1, col=1)

    fig.update_xaxes(title_text="Spanwise Position [m]", row=len(Loads.keys()), col=1)
    fig.update_layout(
        title="Internal Load (NVM) Diagram",
        width=1100,
        height=600,
        template="plotly_dark")

    return fig

def Deflections(x0, x1, pi, **defs):
    titles = [k + f' [{"millimeters" if k[0].lower() in ["w", "v"] else "degrees"}]' for k in defs]
    fig = make_subplots(rows=len(list(defs.keys())), cols=1, shared_xaxes=True, vertical_spacing=0.01)
    xs = linspace(x0, x1, 5000)
    for i, (li, load) in enumerate(zip(defs.keys(), defs.values())):
        fig.append_trace(go.Scatter(
            x=xs,
            y=[load(xi)*(180 / pi if 'degrees' in titles[i] else 1e3) for xi in xs],
            name=li,
        ), row=i+1, col=1)
        fig.update_yaxes(title_text=titles[i], row=i+1, col=1)

    fig.update_xaxes(title_text="Spanwise Position [meters]", row=len(defs.keys()), col=1)
    fig.update_layout(
        title="Deflection Diagram")

    return fig

def DrawFatigue(t, y):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.append_trace(go.Scatter(
        x=t,
        y=y,
        name='Fatigue Cycle',
    ), row=1, col=1)
    fig.update_yaxes(title_text='Stress [MPa]', row=1, col=1)

    fig.update_xaxes(title_text="Time [hours]", row=1, col=1)
    fig.update_layout(title='Fatigue Cycle')
    return fig
