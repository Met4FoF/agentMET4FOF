import plotly.offline as py
import plotly.tools as tls

import matplotlib.pyplot as plt
import numpy as np

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#------------prepare data------------
x = np.arange(0.1, 4, 0.5)
y = np.exp(-x)

plt.figure()
plt.plot(x, y, '-')
plt.title("Simplest errorbars, 0.04 in x, 0.04 in y")

fig = plt.gcf()
plotly_fig = tls.mpl_to_plotly( fig )

plotly_fig["data"][0]["error_y"].update({
                                         "visible": True,
                                         "color":"rgb(255,127,14)",
                                         "value":0.04,
                                         "type":"constant"
                                       })
plotly_fig["data"][0]["error_x"].update({
                                         "visible": True,
                                         "color":"rgb(255,127,14)",
                                         "value":0.04,
                                         "type":"constant"
                                       })

# ---------show legend---------

plt.figure()
data_ = pd.read_excel("harmonized_processed_data.xlsx")

plt.plot(data_['Density'])
plt.plot(data_['ActualTemp'])
fig = plt.gcf()
plotly_fig = tls.mpl_to_plotly( fig )
plotly_fig['layout']['showlegend'] = True

# -----------------------------
# Fixing random state for reproducibility
np.random.seed(19680801)

dt = 0.01
t = np.arange(0, 30, dt)
nse1 = np.random.randn(len(t))                 # white noise 1
nse2 = np.random.randn(len(t))                 # white noise 2

# Two signals with a coherent part at 10Hz and a random part
s1 = np.sin(2 * np.pi * 10 * t) + nse1
s2 = np.sin(2 * np.pi * 10 * t) + nse2

fig, axs = plt.subplots(2, 1)
axs[0].plot(t, s1, t, s2)
axs[0].set_xlim(0, 2)
axs[0].set_xlabel('time')
axs[0].set_ylabel('s1 and s2')
axs[0].grid(True)

cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
axs[1].set_ylabel('coherence')

#convert to plotly
fig.tight_layout()
plotly_fig = tls.mpl_to_plotly( fig )
plotly_fig['layout']['showlegend'] = True

# ------------app layout------------
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=plotly_fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)



