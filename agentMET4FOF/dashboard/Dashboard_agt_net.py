import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
from . import LayoutHelper
from .LayoutHelper import create_nodes_cytoscape, create_edges_cytoscape, \
    create_monitor_graph
from dash.exceptions import PreventUpdate
import networkx as nx

from .Dashboard_layout_base import Dashboard_Layout_Base

class Dashboard_agt_net(Dashboard_Layout_Base):
    def set_layout_name(self,id="agt-net", title="Agent Network"):
        self.id = id
        self.title=title

    def get_multiple_graphs(self,num_monitors=10):
        return [dcc.Graph(id='monitors-graph-'+str(i), figure={},style={'height':'90vh'}) for i in range(num_monitors)]


    def get_layout(self, update_interval_seconds=3, num_monitors=10):
       #body
       return html.Div(className="row",children=[

                    #main panel
                    html.Div(className="col s9", children=[
                            html.Div(className="card", children=[
                               html.Div(className="card-content", children=[
                                       # html.Span(className="card-title", children=["Agent Network"]),
                                       # html.P(children=["Active agents running in agent network"]),
                                       html.Div(className="row", children = [
                                                    html.Div(className="col", children=[
                                                        LayoutHelper.html_button(icon="play_circle_filled",text="Start", id="start-button")
                                                    ]),
                                                    html.Div(className="col", children=[
                                                        LayoutHelper.html_button(icon="stop",text="Stop", id="stop-button")
                                                    ]),

                                                    html.Div(className="col", children=[
                                                        LayoutHelper.html_button(icon="restore",text="Reset", id="reset-button")
                                                    ]),
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

                            html.Div(className="card", id="monitors-temp-division", children=self.get_multiple_graphs(num_monitors))

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

    def prepare_callbacks(self,app):
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
                new_agent = app.dashboard_ctrl.agentNetwork.add_agent(name=type(chosen_dataset).__name__, agentType=agentmet4fof_module.ML_DataStreamAgent)

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
                agent_parameters_texts=[LayoutHelper.visualise_agent_parameters(k,v) for k,v in agent_parameters.items()]
                agent_parameters_div=[html.H6("Parameters: ")] + agent_parameters_texts

                return [current_agent_id, input_names, output_names,agent_parameters_div]
            else:
                return ["Not selected", input_names, output_names, agent_parameters_div]


        #define maximum number of monitors graph
        output_figures = [dash.dependencies.Output('monitors-graph-'+str(i), 'figure') for i in range(app.num_monitors)]
        output_styles = [dash.dependencies.Output('monitors-graph-'+str(i), 'style') for i in range(app.num_monitors)]
        outputs = output_figures+output_styles
        @app.callback(outputs,
                      [dash.dependencies.Input('interval-update-monitor-graph', 'n_intervals')])
        def plot_monitor_memory(n_interval):
            # get nameserver
            agentNetwork = app.dashboard_ctrl.agentNetwork

            # check if agent network is running and first_time running
            # if it isn't, abort updating graphs
            if agentNetwork._get_mode() != "Running" and agentNetwork._get_mode() != "Reset" and n_interval > 0:
                raise PreventUpdate

            agent_names = agentNetwork.agents('MonitorAgent') # get all agent names
            app.num_monitor = len(agent_names)
            monitor_graphs = [{'data': []} for i in range(app.num_monitors)]
            style_graphs = [{'opacity':0, 'width':10,'height':10} for i in range(app.num_monitors)]

            for monitor_id, monitor_agent in enumerate(agent_names):
                memory_data = agentNetwork.get_agent(monitor_agent).get_attr('memory')
                custom_plot_function = agentNetwork.get_agent(monitor_agent).get_attr('custom_plot_function')
                data =[]
                for sender_agent in memory_data.keys():
                    #if custom plot function is not provided, resolve to default plotting
                    if type(custom_plot_function).__name__ == "int":
                        if type(memory_data[sender_agent]) == dict:
                            for attribute in memory_data[sender_agent].keys():
                                data.append(create_monitor_graph(memory_data[sender_agent][attribute],sender_agent+':'+attribute))
                        else:
                            data.append(create_monitor_graph(memory_data[sender_agent],sender_agent))
                    #otherwise call custom plot function and load up custom plot parameters
                    else:
                        custom_plot_parameters = agentNetwork.get_agent(monitor_agent).get_attr('custom_plot_parameters')
                        data.append(custom_plot_function(memory_data[sender_agent],sender_agent,**custom_plot_parameters))
                if len(data) > 5:
                    y_title_offset = 0.1
                else:
                    y_title_offset = -0.1
                monitor_graph={
                    'data': data,
                    'layout': {
                        'title': {
                            'text': monitor_agent,
                            'y':y_title_offset,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'bottom'
                        },
                        'uirevision': app.num_monitor,
                        'showlegend': True,
                        'legend':dict(xanchor='auto',yanchor='bottom', x=1, y=1,orientation= "h"),
                        # 'margin':dict(t=150)
                    },
                }

                monitor_graphs[monitor_id]= monitor_graph
                # style_graphs[monitor_id]= {'opacity':1.0, 'width':'100%','height':'100%'}
                style_graphs[monitor_id]= {'opacity':1.0, 'height':'auto'}
            # monitor_graphs = monitor_graphs+ [{'displayModeBar': False, 'editable': False, 'scrollZoom':False}]
            return monitor_graphs+ style_graphs

        def _handle_matplotlib_figure(input_data, from_agent_name: str):
            """
            Internal function. Checks the mode of matplotlib.figure.Fig to be plotted
            Either it is a base64 str image, or a plotly graph

            This is used in plotting the received matplotlib figures in the MonitorAgent's plot memory.
            """

            if isinstance(input_data, str):
                new_graph = html.Img(src=input_data, title=from_agent_name)
            else:
                new_graph = dcc.Graph(figure=input_data)
            return new_graph
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
                    # get the graph relevant to 'monitor_agent_input'
                    graph = plot_data[from_agent_name]

                    if isinstance(graph, dict):
                        for graph_ in graph.values():
                            new_graph = _handle_matplotlib_figure(graph_, from_agent_name)
                            html_div_monitor.append(new_graph)
                    else:
                        new_graph = _handle_matplotlib_figure(graph, from_agent_name)
                        html_div_monitor.append(new_graph)

                #only add the graph if there is some plots in the Monitor Agent
                if len(html_div_monitor) > 1:
                    all_graphs.append(html.Div(className="card",children=html_div_monitor))

            # set dimensions of each monitor agent's graph
            return [all_graphs]
        return app
