# -*- coding: utf-8 -*-
import os
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc

from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots

import networkx as nx
import agentMET4FOF.agents as agentmet4fof_module

import agentMET4FOF.dashboard.LayoutHelper as LayoutHelper
from agentMET4FOF.dashboard.LayoutHelper import create_nodes_cytoscape, create_edges_cytoscape, create_monitor_graph
from agentMET4FOF.dashboard.Dashboard_ml_exp import get_ml_exp_layout, prepare_ml_exp_callbacks, get_experiments_list
from agentMET4FOF.dashboard.Dashboard_agt_net import get_agt_net_layout, prepare_agt_net_callbacks

external_stylesheets = ['https://fonts.googleapis.com/icon?family=Material+Icons', 'https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css']
external_scripts = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js']

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts
                )


def init_app_layout(app,update_interval_seconds=3,num_monitors=10):
    app.update_interval_seconds = update_interval_seconds
    app.num_monitors = num_monitors
    app.layout = html.Div(children=[
            #header
            html.Nav([
                html.Div([
                    html.A("Met4FoF Agent Testbed", className="brand-logo center"),
                    html.Ul([
                    ], className="right hide-on-med-and-down")
                ], className="nav-wrapper container")
            ], className="light-blue lighten-1"),
            dcc.Tabs(id="main-tabs", value="agt-net", children=[
                dcc.Tab(id="agt-net-tab", value="agt-net",label='Agent Network', children=[
                    get_agt_net_layout(update_interval_seconds,num_monitors)
                ]),
                dcc.Tab(id="ml-exp-tab",value="ml-exp", label='ML Experiments',  children=[
                    get_ml_exp_layout()
                ]),
            ]),
            # html.Div(get_agt_net_layout(update_interval_seconds,num_monitors))
            dcc.Interval(
                id='interval-first-load',
                interval=1000 * 1000,  # in milliseconds
                n_intervals=0
            )
    ])
    prepare_agt_net_callbacks(app)
    prepare_ml_exp_callbacks(app)


    @app.callback([dash.dependencies.Output('agt-net-tab', 'children'),
                  dash.dependencies.Output('ml-exp-tab', 'children')],
                  [dash.dependencies.Input('main-tabs', 'value'),
                   dash.dependencies.Input('interval-first-load', 'n_intervals')
                   ])
    def render_content(tab,intervals):
        if tab == 'ml-exp':
            experiments_df = get_experiments_list()
            return ["",get_ml_exp_layout(experiments_df)]
        else:

            return [get_agt_net_layout(app.update_interval_seconds,app.num_monitors),""]

    return app
