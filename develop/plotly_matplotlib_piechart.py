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

sensor_percentage = [75, 38, 79, 77, 6, 6, 6, 71, 51, 45, 46]
labels = ('Microphone','Vibration plain bearing','Vibration piston rod','Vibration ball bearing', 'Axial force','Pressure','Velocity','Active current','Motor current phase 1','Motor current phase 2','Motor current phase 3')
figsize = (8,8)

fig, ax1 = plt.subplots(figsize=figsize)
ax1.set_title("Percentages of features from each sensor")
ax1.pie(sensor_percentage, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, )
ax1.axis('equal')
# plt.show()

#convert to plotly
fig.tight_layout()
plotly_fig = tls.mpl_to_plotly(fig)
plotly_fig['layout']['showlegend'] = True


# # ------------app layout------------
# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),
#
#     html.Div(children='''
#         Dash: A web application framework for Python.
#     '''),
#
#     dcc.Graph(
#         id='example-graph',
#         figure=plotly_fig
#     )
# ])
#
# if __name__ == '__main__':
#     app.run_server(debug=True)



