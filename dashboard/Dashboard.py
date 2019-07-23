# -*- coding: utf-8 -*-
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from plotly import tools

import numpy as np
import networkx as nx

from AgentMET4FOF import AgentMET4FOF, AgentNetwork
import AgentMET4FOF as agentmet4fof_module
import dashboard.LayoutHelper as LayoutHelper
from dashboard.LayoutHelper import get_nodes, get_edges, create_nodes, create_edges, create_monitor_graph


#external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css', 'https://fonts.googleapis.com/icon?family=Material+Icons']
external_scripts = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js']
app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts
                )
update_interval_seconds = 3

app.layout = html.Div(children=[

    #header
    html.Nav([
        html.Div([
            html.A("Met4FoF Agent Testbed", className="brand-logo"),
            html.Ul([
            ], className="right hide-on-med-and-down")
        ], className="nav-wrapper container")
    ], className="light-blue lighten-1"),

    #body
    html.Div(className="row",children=[

        #main panel
        html.Div(className="col s9", children=[
                html.Div(className="card", children=[
                   html.Div(className="card-content", children=[
                           html.Span(className="card-title", children=["Agent Network"]),
                           html.P(children=["Active agents running in local host"]),
                           html.Div(className="row", children = [
                                        html.Div(className="col", children=[
                                            LayoutHelper.html_button(icon="play_circle_filled",text="Start", id="start-button")
                                        ]),
                                        html.Div(className="col", children=[
                                            LayoutHelper.html_button(icon="stop",text="Stop", id="stop-button")
                                        ])
                                    ])

                    ]),
                   html.Div(className="card-action", children=[
                       cyto.Cytoscape(
                           id='agents-network',
                           layout={'name': 'breadthfirst'},
                           style={'width': '100%', 'height': '400px'},
                           elements=[{'data': {'id': 'add1', 'label': 'add1'}, 'position': {'x': 75, 'y': 75}},
                                     {'data': {'id': 'subt1', 'label': 'subt1'}, 'position': {'x': 75, 'y': 90}},
                                     {'data': {'id': 'subt2', 'label': 'subt2'}, 'position': {'x': 75, 'y': 105}}],
                           stylesheet=[
                                        { 'selector': 'node', 'style':
                                            { 'label': 'data(id)' ,
                                              'shape': 'rectangle' }
                                             },
                                        { 'selector': 'edge',
                                          'style': { 'mid-target-arrow-shape': 'triangle','arrow-scale': 3},
                                        }
                                      ]
                       )

                    ])

                ]),
                html.H5(className="card", id="matplotlib-division", children=" "),

                html.Div(className="card", id="monitors-temp-division", children=[
                    dcc.Graph(id='monitors-graph',
                        figure=go.Figure(),
                        style={'height': 800},
                    ),
                ])


        ]),

        #side panel
        html.Div(className="col s3 ", children=[
            html.Div(className="card blue lighten-4", children= [
                html.Div(className="card-content", children=[

                    html.Div(style={'margin-top': '20px'}, children=[
                        html.H6(className="black-text", children="Add Agent"),
                        dcc.Dropdown(id="add-modules-dropdown"),
                        LayoutHelper.html_button(icon="add_to_queue",text="Add Agent", id="add-module-button")


                    ]),



                    html.Div(style={'margin-top': '20px'}, children=[
                        html.H6(className="black-text", children="Dataset"),
                        dcc.Dropdown(
                            options=[
                                {'label': 'New York City', 'value': 'NYC'},
                                {'label': 'Montréal', 'value': 'MTL'},
                                {'label': 'San Francisco', 'value': 'SF'}
                            ],
                            value='MTL',
                        )
                    ])

                ])

            ]),

            html.Div(className="card green lighten-4", children=[

                html.Div(className="card-content", children=[
                    # side panel contents here
                    html.H5("Agent Configuration"),
                    html.H5(id='selected-node', children="Not selected", className="flow-text", style={"font-weight": "bold"}),

                    html.H6(id='input-names', children="Not selected", className="flow-text"),
                    html.H6(id='output-names', children="Not selected", className="flow-text"),

                    html.Div(style={'margin-top': '20px'}, children=[
                        html.H6(className="black-text", children="Select Output Module"),
                        dcc.Dropdown(id="connect-modules-dropdown"),
                        LayoutHelper.html_button(icon="link",text="Connect Agent", id="connect-module-button"),
                        LayoutHelper.html_button(icon="highlight_off",text="Disconnect Agent", id="disconnect-module-button")


                    ]),

                    html.P(id="connect_placeholder", className="black-text", children=" "),
                    html.P(id="disconnect_placeholder", className="black-text", children=" "),
                    html.P(id="monitor_placeholder", className="black-text", children=" "),



                ])

            ])


        ]),
        dcc.Interval(
            id='interval-component-network-graph',
            interval=update_interval_seconds * 1000,  # in milliseconds
            n_intervals=0
        ),
        dcc.Interval(
            id='interval-add-module-list',
            interval=1000 * 1000,  # in milliseconds
            n_intervals=0
        ),
        dcc.Interval(
            id='interval-update-monitor-graph',
            interval=update_interval_seconds * 1000,  # in milliseconds
            n_intervals=0
        )
    ])
])

#global variables access via 'dashboard_var'
class Dashboard_Control():
    def __init__(self, ip_addr="127.0.0.1", port=3333, modules = []):
        super(Dashboard_Control, self).__init__()
        self.network_layout = {'name': 'grid'}
        self.current_selected_agent = " "
        self.current_nodes = []
        self.current_edges = []
        # get nameserver
        self.agent_graph = nx.Graph()
        self.agentNetwork = AgentNetwork(ip_addr=ip_addr,port=port,mode="Connect")
        self.modules = [agentmet4fof_module] + modules

    def get_agentTypes(self):
        agentTypes ={}
        for module_ in self.modules:
            agentTypes.update(dict([(name, cls) for name, cls in module_.__dict__.items() if
                               isinstance(cls, type) and cls.__bases__[-1] == AgentMET4FOF]))
        return agentTypes

#Update network graph per interval
@app.callback([dash.dependencies.Output('agents-network', 'elements'),
       #        dash.dependencies.Output('agents-network', 'layout'),
               dash.dependencies.Output('connect-modules-dropdown', 'options') ],
              [dash.dependencies.Input('interval-component-network-graph', 'n_intervals')],
              [dash.dependencies.State('agents-network', 'elements')]
               )
def update_network_graph(n_intervals,graph_elements):

    #get nameserver
    agentNetwork = app.dashboard_ctrl.agentNetwork

    #update node graph
    nodes=get_nodes(agentNetwork)
    edges=get_edges(agentNetwork)

    print(nodes)
    print(app.dashboard_ctrl.agent_graph.nodes())
    #if current number more than before, then update graph
    if(len(app.dashboard_ctrl.current_nodes) != len(nodes)) or (len(app.dashboard_ctrl.current_edges) != len(edges)) or n_intervals == 0:
    #if (dashboard_ctrl.agent_graph.number_of_nodes() != len(nodes)) or (
    #    dashboard_ctrl.agent_graph.number_of_edges() != len(edges)) or n_intervals == 0:

        app.dashboard_ctrl.agent_graph.add_nodes_from(nodes)
        app.dashboard_ctrl.agent_graph.add_edges_from(edges)

        nodes_elements = create_nodes(app.dashboard_ctrl.agent_graph)
        edges_elements = create_edges(edges)

        # update agents connect options
        node_connect_options = [{'label': agentName, 'value': agentName} for agentName in nodes]

        app.dashboard_ctrl.current_nodes = nodes
        app.dashboard_ctrl.current_edges = edges



       # return [nodes_elements + edges_elements,dashboard_ctrl.network_layout ,node_connect_options]
        return [nodes_elements + edges_elements, node_connect_options]

    else:
        raise PreventUpdate


@app.callback( dash.dependencies.Output('add-modules-dropdown', 'options'),
              [dash.dependencies.Input('interval-add-module-list', 'n_intervals')
               ])
def update_add_module_list(n_interval):
    #get nameserver
    #agentNetwork = dashboard_ctrl.agentNetwork

    #agentTypes = dashboard_ctrl.get_agentTypes()
    #module_add_options = [ {'label': agentType, 'value': agentType} for agentType in list(agentTypes.keys())]
    agentTypes = app.dashboard_ctrl.get_agentTypes()
    module_add_options = [{'label': agentType, 'value': agentType} for agentType in list(agentTypes.keys())]

    return module_add_options

#Start button click
@app.callback( dash.dependencies.Output('start-button', 'children'),
              [dash.dependencies.Input('start-button', 'n_clicks')
               ])
def start_button_click(n_clicks):
    if n_clicks is not None:
        app.dashboard_ctrl.agentNetwork.set_running_state()
    return LayoutHelper.html_icon("play_circle_filled","Start")

#Stop button click
@app.callback( dash.dependencies.Output('stop-button', 'children'),
              [dash.dependencies.Input('stop-button', 'n_clicks')
               ])
def stop_button_click(n_clicks):
    if n_clicks is not None:
        app.dashboard_ctrl.agentNetwork.set_stop_state()
    return LayoutHelper.html_icon("stop","Stop")

#Add agent button click
@app.callback( dash.dependencies.Output('add-module-button', 'children'),
              [dash.dependencies.Input('add-module-button', 'n_clicks')],
              [dash.dependencies.State('add-modules-dropdown', 'value')]
               )
def add_module_button_click(n_clicks,add_dropdown_val):
    #for add agent button click
    if n_clicks is not None:
        agentTypes = app.dashboard_ctrl.get_agentTypes()
        new_agent = app.dashboard_ctrl.agentNetwork.add_agent(agentType=agentTypes[add_dropdown_val])

    return LayoutHelper.html_icon("add_to_queue","Add Agent")

@app.callback(dash.dependencies.Output('connect_placeholder', 'children') ,
              [dash.dependencies.Input('connect-module-button', 'n_clicks')],
              [dash.dependencies.State('connect-modules-dropdown', 'value'),
               dash.dependencies.State('selected-node', 'children')])
def bind_module_click(n_clicks_connect,dropdown_value, current_agent_id):
    if(n_clicks_connect is not None):
        #get nameserver
        agentNetwork = app.dashboard_ctrl.agentNetwork

        print(agentNetwork.get_agent(current_agent_id))
        print(agentNetwork.get_agent(dropdown_value))
        #for connect module button click
        agentNetwork.bind_agents(agentNetwork.get_agent(current_agent_id), agentNetwork.get_agent(dropdown_value))
    return " "

@app.callback(dash.dependencies.Output('disconnect_placeholder', 'children') ,
              [dash.dependencies.Input('disconnect-module-button', 'n_clicks')],
              [dash.dependencies.State('connect-modules-dropdown', 'value'),
               dash.dependencies.State('selected-node', 'children')])
def unbind_module_click(n_clicks_connect,dropdown_value, current_agent_id):
    if(n_clicks_connect is not None):
        #get nameserver
        agentNetwork = app.dashboard_ctrl.agentNetwork

        print(agentNetwork.get_agent(current_agent_id))
        print(agentNetwork.get_agent(dropdown_value))
        #for connect module button click
        agentNetwork.unbind_agents(agentNetwork.get_agent(current_agent_id), agentNetwork.get_agent(dropdown_value))
    return " "

@app.callback([dash.dependencies.Output('selected-node', 'children'),
               dash.dependencies.Output('input-names', 'children'),
               dash.dependencies.Output('output-names', 'children')
               ],
              [dash.dependencies.Input('agents-network', 'tapNodeData')])
def displayTapNodeData(data):
    input_names =[]
    output_names=[]

    if data is not None:
        current_agent_id = data['id']
        #get nameserver
        agentNetwork = app.dashboard_ctrl.agentNetwork

        app.dashboard_ctrl.current_selected_agent = agentNetwork.get_agent(current_agent_id)
        input_names=list(app.dashboard_ctrl.current_selected_agent.get_attr('Inputs').keys())
        output_names=list(app.dashboard_ctrl.current_selected_agent.get_attr('Outputs').keys())

        #formatting
        input_names =["Inputs: "]+[input_name+", "for input_name in input_names]
        output_names = ["Outputs: "] + [output_name + ", " for output_name in output_names]

        return [current_agent_id, input_names, output_names]
    else:
        return ["Not selected", input_names, output_names]

#load Monitors data and draw - all at once
@app.callback( [dash.dependencies.Output('monitors-graph', 'figure'),
                dash.dependencies.Output('monitors-graph', 'style')],
               [dash.dependencies.Input('interval-update-monitor-graph', 'n_intervals')])
def plot_monitor_memory(n_interval):
    # get nameserver
    agentNetwork = app.dashboard_ctrl.agentNetwork

    # check if agent network is running and first_time running
    # if it isn't, abort updating graphs
    print("n_interval-monitor-graph",n_interval)
    if agentNetwork._get_mode() != "Running" and n_interval > 0:
        print(agentNetwork._get_mode())
        raise PreventUpdate

    agent_names = agentNetwork.agents() # get all agent names
    agent_type ="Monitor" #all agents with Monitor in its name will be selected
    monitors_data = {} #storage for all monitor agent's memory

    # load data from all Monitor agent's memory
    for agent_name in agent_names:
        if agent_type in agent_name:
            monitor_agent = agentNetwork.get_agent(agent_name)
            memory = monitor_agent.get_attr('memory')
            monitors_data.update({agent_name:memory})

    # now monitors_data = {'monitor_agent1_name':agent1_memory, 'monitor_agent2_name':agent2_memory }
    # now create a plot for each monitor agent
    # initialize necessary variables for plotting multi-graphs
    subplot_titles = tuple(list(monitors_data.keys()))
    num_graphs = len(list(monitors_data.keys()))
    monitor_graphs = tools.make_subplots(rows=num_graphs, cols=1, subplot_titles=subplot_titles)

    # now loop through monitors_data's every monitor agent's memory
    # build a graph from every agent's memory via create_monitor_graph()
    for count, agent_name in enumerate(monitors_data.keys()):
        monitor_data = monitors_data[agent_name]

        # create a new graph for every 'Input agent' connected to Monitor Agent
        for from_agent_name in monitor_data:
            # get the 'data' relevant to 'monitor_agent_input'
            input_data = monitor_data[from_agent_name]

            # create a graph from it
            # and update the graph's name according to the agent's name
            if type(input_data).__name__ == 'list' or type(input_data).__name__ == 'ndarray':
                monitor_graph = create_monitor_graph(input_data)
                monitor_graph.update(name=from_agent_name)
                # lastly- append this graph into the master graphs list which is monitor_graphs
                monitor_graphs.append_trace(monitor_graph, count+1, 1)
            elif type(input_data).__name__ == 'dict':
                print("LIST: ",from_agent_name,list(input_data.keys()))
                print(input_data)
                for channel in input_data.keys():
                    monitor_graph = create_monitor_graph(input_data[channel])
                    monitor_graph.update(name=from_agent_name+" : "+channel)
                    # lastly- append this graph into the master graphs list which is monitor_graphs
                    monitor_graphs.append_trace(monitor_graph, count+1, 1)

    # set to show legend
    monitor_graphs['layout'].update(showlegend = True)

    # set dimensions of each monitor agent's graph
    constant_height_px= 400
    height_px = num_graphs*constant_height_px
    style = {'height': height_px}
    return [monitor_graphs,style]

#load Monitors data and draw - all at once
@app.callback([dash.dependencies.Output('matplotlib-division', 'children')],
              [dash.dependencies.Input('interval-update-monitor-graph', 'n_intervals')])
def plot_monitor_graphs(n_interval):
    # get nameserver
    agentNetwork = app.dashboard_ctrl.agentNetwork

    # check if agent network is running and first_time running
    # if it isn't, abort updating graphs
    print("n_interval-monitor-graph",n_interval)
    if agentNetwork._get_mode() != "Running" and n_interval > 0:
        print(agentNetwork._get_mode())
        raise PreventUpdate

    agent_names = agentNetwork.agents() # get all agent names
    agent_type ="Monitor" #all agents with Monitor in its name will be selected
    plots_data = {} #storage for all monitor agent's memory

    # load data from all Monitor agent's memory
    for agent_name in agent_names:
        if agent_type in agent_name:
            monitor_agent = agentNetwork.get_agent(agent_name)
            plots = monitor_agent.get_attr('plots')
            plots_data.update({agent_name:plots})

    # now monitors_data = {'monitor_agent1_name':agent1_memory, 'monitor_agent2_name':agent2_memory }
    # now create a plot for each monitor agent
    # initialize necessary variables for plotting multi-graphs
    subplot_titles = tuple(list(plots_data.keys()))
    num_graphs = len(list(plots_data.keys()))
    all_graphs = []
    # now loop through monitors_data's every monitor agent's memory
    # build a graph from every monitor agent's `plots`
    for count, agent_name in enumerate(plots_data.keys()):
        plot_data = plots_data[agent_name]
        html_div_monitor =[]
        html_div_monitor.append(html.H5(agent_name, style={"text-align": "center"}))
        # create a new graph for every agent
        for from_agent_name in plot_data:
            # get the 'data' relevant to 'monitor_agent_input'
            input_data = plot_data[from_agent_name]
            print(input_data)

            #new_graph = dcc.Graph(figure=input_data)
            from_agent_title = html.Figcaption(from_agent_name)
            new_graph = html.Img(src=input_data, title=from_agent_name)

            # html_div_monitor.append(html.Div(children=[new_graph,from_agent_title]))
            # html_div_monitor.append(from_agent_title)
            html_div_monitor.append(new_graph)

        #only add the graph if there is some plots in the Monitor Agent
        if len(html_div_monitor) > 1:
            all_graphs.append(html.Div(className="card",children=html_div_monitor))

    # set dimensions of each monitor agent's graph
    print(all_graphs)
    return [all_graphs]


