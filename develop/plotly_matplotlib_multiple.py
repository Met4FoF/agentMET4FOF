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


#convert matplotlib to plotly
import matplotlib.pyplot as plt
from plotly import tools as tls

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

#convert plotly
fig.tight_layout()
plotly_fig = tls.mpl_to_plotly(fig)
plotly_fig['layout']['showlegend'] = True
print(plotly_fig)
#===========native plotly======
import plotly.graph_objs as go
native_trace = go.Scatter(x=[1, 2, 3], y=[4, 5, 6])
#native_trace_figure = go.Figure(native_trace)
#============append graph=========
monitor_graphs = tls.make_subplots(rows=4, cols=1)
#monitor_graphs = go.Figure()
# # monitor_graphs.append_trace(plotly_fig['data'][0], 2, 1)
# # monitor_graphs.append_trace(plotly_fig['data'][1], 2, 1)
# # monitor_graphs.append_trace(plotly_fig['data'][2], 2, 1)
# #monitor_graphs.layout.update(plotly_fig['layout'][0])
# monitor_graphs.add_traces([plotly_fig.data[0],plotly_fig.data[1],plotly_fig.data[2]])
# #monitor_graphs.layout.update(native_trace_figure['layout'])
# monitor_graphs.layout.update(plotly_fig.layout)
# monitor_graphs.layout.update(plotly_fig.layout)
#
# monitor_graphs

monitor_graphs.append_trace(plotly_fig.data[0],1,1)
monitor_graphs.append_trace(plotly_fig.data[1],2,1)
monitor_graphs.append_trace(plotly_fig.data[2],3,1)

#monitor_graphs.layout.update(plotly_fig.layout)

# f_widget1 = go.FigureWidget(plotly_fig)
# f_widget2 = go.FigureWidget(plotly_fig)
#
# from ipywidgets import HBox
#
# fwid = HBox([f_widget1,f_widget2])
#
# from IPython.display import display
#
# display(fwid)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# ------------app layout------------
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    html.Div([



    ])
    dcc.Graph(
        id='example-graph',
        #figure=plotly_fig
        figure = monitor_graphs
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)



