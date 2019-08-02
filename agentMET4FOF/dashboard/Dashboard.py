# -*- coding: utf-8 -*-
import os
import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from plotly import tools

import networkx as nx
import agentMET4FOF.agents as agentmet4fof_module

import agentMET4FOF.dashboard.LayoutHelper as LayoutHelper
from agentMET4FOF.dashboard.LayoutHelper import create_nodes_cytoscape, create_edges_cytoscape, create_monitor_graph

external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css', 'https://fonts.googleapis.com/icon?family=Material+Icons']
external_scripts = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js']
assets_url_path = os.path.join(os.path.dirname(__file__), 'assets')

app = dash.Dash(__name__,
                assets_url_path=assets_url_path,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts
                )
app.update_interval_seconds = 3

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
                                        ]),

                                        html.Div(className="col", children=[
                                            LayoutHelper.html_button(icon="restore",text="Reset", id="reset-button")
                                        ])
                                    ])

                    ]),
                   html.Div(className="card-action", children=[
                       cyto.Cytoscape(
                           id='agents-network',
                           layout={'name': 'breadthfirst'},
                           style={'width': '100%', 'height': '600px'},
                           elements=[],
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
                        #style={'height': 800},
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
                        LayoutHelper.html_button(icon="person_add",text="Add Agent", id="add-module-button"),
                        LayoutHelper.html_button(icon="delete_forever",text="Remove Agent", id="remove-module-button", style={"margin-left":'4px'})

                    ]),

                    html.Div(style={'margin-top': '20px'}, children=[
                        html.H6(className="black-text", children="Dataset"),
                        dcc.Dropdown(id="add-dataset-dropdown"),
                        LayoutHelper.html_button(icon="add_to_queue",text="Add Datastream Agent", id="add-dataset-button")
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
                        LayoutHelper.html_button(icon="highlight_off",text="Disconnect Agent", id="disconnect-module-button", style={"margin-left":'4px'})


                    ]),
                    html.H6(id='agent-parameters', children="Not selected", className="flow-text"),

                    html.P(id="connect_placeholder", className="black-text", children=" "),
                    html.P(id="disconnect_placeholder", className="black-text", children=" "),
                    html.P(id="monitor_placeholder", className="black-text", children=" "),



                ])

            ])


        ]),
        dcc.Interval(
            id='interval-component-network-graph',
            interval=app.update_interval_seconds * 1000,  # in milliseconds
            n_intervals=0
        ),
        dcc.Interval(
            id='interval-add-module-list',
            interval=1000 * 1000,  # in milliseconds
            n_intervals=0
        ),
        dcc.Interval(
            id='interval-update-monitor-graph',
            interval=app.update_interval_seconds * 1000,  # in milliseconds
            n_intervals=0
        )
    ])
])

#Update network graph per interval
@app.callback([dash.dependencies.Output('agents-network', 'elements'),
               dash.dependencies.Output('connect-modules-dropdown', 'options')],
              [dash.dependencies.Input('interval-component-network-graph', 'n_intervals')],
              [dash.dependencies.State('agents-network', 'elements')]
               )
def update_network_graph(n_intervals,graph_elements):

    #get nameserver
    agentNetwork = app.dashboard_ctrl.agentNetwork

    #update node graph
    agentNetwork.update_networkx()
    nodes, edges = agentNetwork.get_nodes_edges()

    #if current graph number is different more than before, then update graph
    if len(graph_elements) != (len(nodes) + len(edges)):
    #if(app.dashboard_ctrl.agent_graph.number_of_nodes() != len(nodes) or app.dashboard_ctrl.agent_graph.number_of_edges() != len(edges)) or n_intervals == 0:
        new_G = nx.DiGraph()
        new_G.add_nodes_from(nodes)
        new_G.add_edges_from(edges)

        nodes_elements = create_nodes_cytoscape(new_G)
        edges_elements = create_edges_cytoscape(edges)

        app.dashboard_ctrl.agent_graph = new_G

        # update agents connect options
        node_connect_options = [{'label': agentName, 'value': agentName} for agentName in nodes]

        return [nodes_elements + edges_elements, node_connect_options]

    else:
        raise PreventUpdate



@app.callback( [dash.dependencies.Output('add-modules-dropdown', 'options'),
                dash.dependencies.Output('add-dataset-dropdown', 'options')
                ],
              [dash.dependencies.Input('interval-add-module-list', 'n_intervals')
               ])
def update_add_module_list(n_interval):
    #get nameserver
    #agentNetwork = dashboard_ctrl.agentNetwork

    #agentTypes = dashboard_ctrl.get_agentTypes()
    #module_add_options = [ {'label': agentType, 'value': agentType} for agentType in list(agentTypes.keys())]
    agentTypes = app.dashboard_ctrl.get_agentTypes()
    module_add_options = [{'label': agentType, 'value': agentType} for agentType in list(agentTypes.keys())]

    datasets = app.dashboard_ctrl.get_datasets()
    module_dataset_options = [{'label': dataset, 'value': dataset} for dataset in list(datasets.keys())]

    return [module_add_options,module_dataset_options]

#Start button click
@app.callback( dash.dependencies.Output('start-button', 'children'),
              [dash.dependencies.Input('start-button', 'n_clicks')
               ])
def start_button_click(n_clicks):
    if n_clicks is not None:
        app.dashboard_ctrl.agentNetwork.set_running_state()
    raise PreventUpdate

#Stop button click
@app.callback( dash.dependencies.Output('stop-button', 'children'),
              [dash.dependencies.Input('stop-button', 'n_clicks')
               ])
def stop_button_click(n_clicks):
    if n_clicks is not None:
        app.dashboard_ctrl.agentNetwork.set_stop_state()
    raise PreventUpdate

#Stop button click
@app.callback( dash.dependencies.Output('reset-button', 'children'),
              [dash.dependencies.Input('reset-button', 'n_clicks')
               ])
def stop_button_click(n_clicks):
    if n_clicks is not None:
        app.dashboard_ctrl.agentNetwork.reset_agents()
    raise PreventUpdate

#Add agent button click
@app.callback( dash.dependencies.Output('add-module-button', 'children'),
              [dash.dependencies.Input('add-module-button', 'n_clicks')],
              [dash.dependencies.State('add-modules-dropdown', 'value')]
               )
def add_agent_button_click(n_clicks,add_dropdown_val):
    #for add agent button click
    if n_clicks is not None:
        agentTypes = app.dashboard_ctrl.get_agentTypes()
        new_agent = app.dashboard_ctrl.agentNetwork.add_agent(agentType=agentTypes[add_dropdown_val])
    raise PreventUpdate

#Add agent button click
@app.callback( dash.dependencies.Output('remove-module-button', 'children'),
              [dash.dependencies.Input('remove-module-button', 'n_clicks')],
              [dash.dependencies.State('selected-node', 'children')]
               )
def remove_agent_button_click(n_clicks,current_agent_id):
    #for add agent button click
    if n_clicks is not None and current_agent_id != "Not selected":
        app.dashboard_ctrl.agentNetwork.remove_agent(current_agent_id)
    raise PreventUpdate

#Add agent button click
@app.callback( dash.dependencies.Output('add-dataset-button', 'children'),
              [dash.dependencies.Input('add-dataset-button', 'n_clicks')],
              [dash.dependencies.State('add-dataset-dropdown', 'value')]
               )
def add_dataset_button_click(n_clicks,add_dropdown_val):
    #for add agent button click
    if n_clicks is not None:
        agentTypes = app.dashboard_ctrl.get_agentTypes()
        datasets = app.dashboard_ctrl.get_datasets()
        chosen_dataset = datasets[add_dropdown_val]()
        new_agent = app.dashboard_ctrl.agentNetwork.add_agent(name=type(chosen_dataset).__name__,agentType=agentmet4fof_module.DataStreamAgent)

        new_agent.init_parameters(stream=chosen_dataset)
    raise PreventUpdate

@app.callback(dash.dependencies.Output('connect_placeholder', 'children') ,
              [dash.dependencies.Input('connect-module-button', 'n_clicks')],
              [dash.dependencies.State('connect-modules-dropdown', 'value'),
               dash.dependencies.State('selected-node', 'children')])
def bind_module_click(n_clicks_connect,dropdown_value, current_agent_id):
    if n_clicks_connect is not None and current_agent_id != "Not selected":
        #get nameserver
        agentNetwork = app.dashboard_ctrl.agentNetwork

        #for connect module button click
        if current_agent_id != dropdown_value:
            agentNetwork.bind_agents(agentNetwork.get_agent(current_agent_id), agentNetwork.get_agent(dropdown_value))
    raise PreventUpdate

@app.callback(dash.dependencies.Output('disconnect_placeholder', 'children') ,
              [dash.dependencies.Input('disconnect-module-button', 'n_clicks')],
              [dash.dependencies.State('connect-modules-dropdown', 'value'),
               dash.dependencies.State('selected-node', 'children')])
def unbind_module_click(n_clicks_connect,dropdown_value, current_agent_id):
    if(n_clicks_connect is not None):
        #get nameserver
        agentNetwork = app.dashboard_ctrl.agentNetwork

        #for connect module button click
        agentNetwork.unbind_agents(agentNetwork.get_agent(current_agent_id), agentNetwork.get_agent(dropdown_value))
    raise PreventUpdate

@app.callback([dash.dependencies.Output('selected-node', 'children'),
               dash.dependencies.Output('input-names', 'children'),
               dash.dependencies.Output('output-names', 'children'),
               dash.dependencies.Output('agent-parameters', 'children'),
               ],
              [dash.dependencies.Input('agents-network', 'tapNodeData')])
def displayTapNodeData(data):
    input_names =[]
    output_names=[]
    agent_parameters_div = []

    if data is not None:
        current_agent_id = data['id']
        #get nameserver
        agentNetwork = app.dashboard_ctrl.agentNetwork

        app.dashboard_ctrl.current_selected_agent = agentNetwork.get_agent(current_agent_id)
        input_names=list(app.dashboard_ctrl.current_selected_agent.get_attr('Inputs').keys())
        output_names=list(app.dashboard_ctrl.current_selected_agent.get_attr('Outputs').keys())
        agent_parameters = app.dashboard_ctrl.current_selected_agent.get_all_attr()

        #formatting
        input_names =["Inputs: "]+[input_name+", "for input_name in input_names]
        output_names = ["Outputs: "] + [output_name + ", " for output_name in output_names]
        agent_parameters_texts=[html.H6(k +": "+str(v)) for k,v in agent_parameters.items()]
        agent_parameters_div=[html.H6("Parameters: ")] + agent_parameters_texts

        return [current_agent_id, input_names, output_names,agent_parameters_div]
    else:
        return ["Not selected", input_names, output_names, agent_parameters_div]


#load Monitors data and draw - all at once
@app.callback( [dash.dependencies.Output('monitors-graph', 'figure')],
             #   dash.dependencies.Output('monitors-graph', 'style')],
               [dash.dependencies.Input('interval-update-monitor-graph', 'n_intervals')])
def plot_monitor_memory(n_interval):
    # get nameserver
    agentNetwork = app.dashboard_ctrl.agentNetwork

    # check if agent network is running and first_time running
    # if it isn't, abort updating graphs
    if agentNetwork._get_mode() != "Running" and agentNetwork._get_mode() != "Reset" and n_interval > 0:
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
    if num_graphs > 0:
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
        return [monitor_graphs]
    else:
        return [[]]

#load Monitors data and draw - all at once
@app.callback([dash.dependencies.Output('matplotlib-division', 'children')],
              [dash.dependencies.Input('interval-update-monitor-graph', 'n_intervals')])
def plot_monitor_graphs(n_interval):
    # get nameserver
    agentNetwork = app.dashboard_ctrl.agentNetwork

    # check if agent network is running and first_time running
    # if it isn't, abort updating graphs
    if agentNetwork._get_mode() != "Running" and n_interval > 0:
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

            if type(input_data).__name__ == 'dict':
                for key in input_data.keys():
                    new_graph = html.Img(src=input_data[key], title=from_agent_name)
                    html_div_monitor.append(new_graph)
            else:
                new_graph = html.Img(src=input_data, title=from_agent_name)
                html_div_monitor.append(new_graph)

        #only add the graph if there is some plots in the Monitor Agent
        if len(html_div_monitor) > 1:
            all_graphs.append(html.Div(className="card",children=html_div_monitor))

    # set dimensions of each monitor agent's graph
    return [all_graphs]


