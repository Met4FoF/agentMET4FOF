import warnings

import dash
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html
import networkx as nx
import visdcc

from dash.dependencies import ClientsideFunction
from dash.exceptions import PreventUpdate

from . import LayoutHelper
from .Dashboard_layout_base import Dashboard_Layout_Base
from .LayoutHelper import (
    create_edges_cytoscape,
    create_monitor_graph,
    create_nodes_cytoscape,
    extract_param_dropdown,
    get_param_dash_component,
)
from .. import agents as agentmet4fof_module

class Dashboard_agt_net(Dashboard_Layout_Base):
    def set_layout_name(self, id="agt-net", title="Agent Network"):
        self.id = id
        self.title = title

    def get_multiple_graphs(self, num_monitors=10):
        return [dcc.Graph(id='monitors-graph-' + str(i), figure={}, style={'height': '90vh'}) for i in
                range(num_monitors)]

    def get_layout(self):
        # body
        return html.Div(className="row", children=[

            # main panel
            html.Div(className="col s9", children=[
                html.Div(className="card", children=[
                    html.Div(className="card-content", children=[
                        # html.Span(className="card-title", children=["Agent Network"]),
                        # html.P(children=["Active agents running in agent network"]),
                        html.Div(className="row", children=[
                            html.Div(className="col", children=[
                                LayoutHelper.html_button(icon="play_circle_filled", text="Start", id="start-button")
                            ]),
                            html.Div(className="col", children=[
                                LayoutHelper.html_button(icon="stop", text="Stop", id="stop-button")
                            ]),

                            html.Div(className="col", children=[
                                LayoutHelper.html_button(icon="restore", text="Reset", id="reset-button")
                            ]),
                            html.Div(className="col", children=[
                                LayoutHelper.html_button(icon="restore", text="Export JPG", id="cyto-button-export")
                            ]),
                        ])

                    ]),
                    html.Div(className="card-action", children=[
                        cyto.Cytoscape(
                            id='agents-network',
                            layout={'name': 'breadthfirst'},
                            style={'width': '100%', 'height': '800px'},
                            elements=[],
                            stylesheet=self.app.network_stylesheet,

                        )

                    ])

                ]),
                html.H5(className="card", id="matplotlib-division", children=" "),
                html.Div(className="card", id="monitors-temp-division",
                         children=self.get_multiple_graphs(self.app.num_monitors)),

            ]),

            # side panel
            html.Div(className="col s3 ", children=[
                html.Div(className="card blue lighten-4", children=[
                    html.Div(className="card-content", children=[

                        html.Div(style={'margin-top': '20px'}, children=[
                            html.H6(className="black-text", children="Add Agent"),
                            dcc.Dropdown(id="add-modules-dropdown", style={'margin-bottom': '35px'}),

                            html.Div(className="input-field", id="agent-init-div", children=[
                                html.I(className="material-icons prefix", children=["account_circle"]),
                                dcc.Input(
                                    id="agent-init-name",
                                    type="text",
                                    value="",
                                    className="validate"
                                ),
                                html.Label(children="New Agent Name", htmlFor="agent-init-name", className="active")
                            ]),
                            html.Div(style={'margin-top': '10px'}, id="agent-init-params", children=[

                            ]),
                            LayoutHelper.html_button(icon="person_add", text="Add Agent", id="add-module-button"),
                            LayoutHelper.html_button(icon="delete_forever", text="Remove Agent",
                                                     id="remove-module-button", style={"margin-left": '4px'})

                        ]),

                        html.Div(style={'margin-top': '30px'}, children=[
                            html.H6(className="black-text", children="Add Coalition", style={'margin-bottom': '20px'}),
                            html.Div(className="input-field", children=[
                                html.I(className="material-icons prefix", children=["dvr"]),
                                dcc.Input(
                                    id="coalition-name",
                                    type="text",
                                    value="",
                                    className="validate"
                                ),
                                html.Label(children="New Coalition Name", htmlFor="coalition-name")
                            ]),
                            LayoutHelper.html_button(icon="add_to_queue", text="Add Coalition",
                                                     id="add-coalition-button")
                        ])
                    ])
                ]),

                html.Div(className="card green lighten-4", children=[

                    html.Div(className="card-content", children=[
                        # side panel contents here
                        html.H5("Agent Configuration"),
                        html.H6(id='selected-node', children="Not selected", className="flow-text",
                                style={"font-weight": "bold"}),

                        html.H6(id='input-names', children="Not selected", className="flow-text"),
                        html.H6(id='output-names', children="Not selected", className="flow-text"),

                        html.Div(style={'margin-top': '20px'}, children=[
                            html.H6(className="black-text", children="Select Output Module"),
                            dcc.Dropdown(id="connect-modules-dropdown"),

                            LayoutHelper.html_button(icon="link", text="Connect Agent", id="connect-module-button"),
                            LayoutHelper.html_button(icon="highlight_off", text="Disconnect Agent",
                                                     id="disconnect-module-button", style={"margin-left": '4px'})

                        ]),
                        html.H6(id='agent-parameters', children="Not selected", className="flow-text"),
                        html.P(id="connect_placeholder", className="black-text", children=" "),
                        html.P(id="disconnect_placeholder", className="black-text", children=" "),
                        html.P(id="monitor_placeholder", className="black-text", children=" "),
                        html.P(id="mpld3_placeholder", className="black-text", children=" "),
                        visdcc.Run_js(id='toast-js-script')
                    ])

                ])

            ]),
            dcc.Interval(
                id='interval-component-network-graph',
                interval=self.app.update_interval_seconds * 1000,  # in milliseconds
                n_intervals=0
            ),
            dcc.Interval(
                id='interval-add-module-list',
                interval=1000 * 1000,  # in milliseconds
                n_intervals=0
            ),
            dcc.Interval(
                id='interval-update-monitor-graph',
                interval=self.app.update_interval_seconds * 1000,  # in milliseconds
                n_intervals=0
            ),
            dcc.Interval(
                id='interval-update-toast',
                interval=1 * 1000,  # in milliseconds
                n_intervals=0
            )
        ])

    def prepare_callbacks(self, app):

        @app.callback([dash.dependencies.Output('agents-network', 'generateImage')],
                       [dash.dependencies.Input('cyto-button-export', 'n_clicks')],
                      )
        def export_image(n_clicks):
            if (n_clicks is not None):
                return [{
                    'type': "png",
                    'action': "download"
                }]
            else:
                raise PreventUpdate

        # Update network graph per interval
        @app.callback([dash.dependencies.Output('agents-network', 'elements'),
                       dash.dependencies.Output('connect-modules-dropdown', 'options')],
                      [dash.dependencies.Input('interval-component-network-graph', 'n_intervals'),
                       # dash.dependencies.Input('connect-module-button', 'n_clicks')
                       ],
                      [dash.dependencies.State('agents-network', 'elements')]
                      )
        def update_network_graph(n_intervals, graph_elements):

            # get nameserver
            agentNetwork = app.dashboard_ctrl.agentNetwork

            # update node graph
            agentNetwork.update_networkx()
            nodes, edges = agentNetwork.get_nodes_edges()

            # get coalitions of agents
            coalitions = agentNetwork.coalitions
            num_agent_coalitions = sum([len(coalition.agent_names()) for coalition in coalitions])
            if not hasattr(app, "num_agent_coalitions"):
                app.num_agent_coalitions = num_agent_coalitions

            # if current graph number is different more than before, then update graph
            if (len(graph_elements) != (len(nodes) + len(edges) + len(coalitions))) or (app.num_agent_coalitions != num_agent_coalitions):
                # if(app.dashboard_ctrl.agent_graph.number_of_nodes() != len(nodes) or app.dashboard_ctrl.agent_graph.number_of_edges() != len(edges)) or n_intervals == 0:
                new_G = nx.DiGraph()
                new_G.add_nodes_from(nodes(data=True))
                new_G.add_edges_from(edges)

                nodes_elements = create_nodes_cytoscape(new_G)
                edges_elements = create_edges_cytoscape(edges, app.hide_default_edge)
                # print(edges_elements)
                # draw coalition nodes, and assign child nodes to coalition nodes
                if len(agentNetwork.coalitions) > 0:
                    parent_elements = [{"data": {'id': coalition.name, 'label': coalition.name},
                                        'classes': 'coalition'} for coalition in agentNetwork.coalitions]
                    for coalition in coalitions:
                        # check if agent is in the coalition, set its parents
                        for agent_node in nodes_elements:
                            if agent_node["data"]["id"] in coalition.agent_names():
                                agent_node["data"].update({'parent': coalition.name})

                        # change edge styles within coalition to dashed
                        for edges in edges_elements:
                            if edges["data"]["source"] in coalition.agent_names() and edges["data"][
                                "target"] in coalition.agent_names():
                                edges.update({'classes': "coalition-edge"})
                else:
                    parent_elements = []

                # update agent graph and number of coalitions
                app.dashboard_ctrl.agent_graph = new_G
                app.num_agent_coalitions = num_agent_coalitions

                # update agents connect options
                node_connect_options = [{'label': agentName, 'value': agentName} for agentName in nodes]

                return [nodes_elements + edges_elements + parent_elements, node_connect_options]

            else:
                raise PreventUpdate

        @app.callback([dash.dependencies.Output('add-modules-dropdown', 'options'),
                       ],
                      [dash.dependencies.Input('interval-add-module-list', 'n_intervals')
                       ])
        def update_add_module_list(n_interval):
            # get nameserver
            agentTypes = app.dashboard_ctrl.get_agentTypes()
            module_add_options = [{'label': agentType, 'value': agentType} for agentType in list(agentTypes.keys())]

            datasets = app.dashboard_ctrl.get_datasets()
            module_dataset_options = [{'label': dataset, 'value': dataset} for dataset in list(datasets.keys())]

            return [module_add_options]

        # Start button click
        @app.callback(dash.dependencies.Output('start-button', 'children'),
                      [dash.dependencies.Input('start-button', 'n_clicks')
                       ])
        def start_button_click(n_clicks):
            if n_clicks is not None:
                app.dashboard_ctrl.agentNetwork.set_running_state()
                raise_toast("Set agents state : %s !" % "Running")
            raise PreventUpdate

        # Stop button click
        @app.callback(dash.dependencies.Output('stop-button', 'children'),
                      [dash.dependencies.Input('stop-button', 'n_clicks')
                       ])
        def stop_button_click(n_clicks):
            if n_clicks is not None:
                app.dashboard_ctrl.agentNetwork.set_stop_state()
                raise_toast("Set agents state : %s !" % "Stop")
            raise PreventUpdate

        # Handle click of the reset button.
        @app.callback(
            dash.dependencies.Output("reset-button", "children"),
            [dash.dependencies.Input("reset-button", "n_clicks")],
        )
        def reset_button_click(n_clicks):
            if n_clicks is not None:
                app.dashboard_ctrl.agentNetwork.reset_agents()
                raise_toast("Reset agent states !")
            raise PreventUpdate


        # Init Agent Parameters choices
        @app.callback([dash.dependencies.Output('agent-init-name', 'value'),
                       dash.dependencies.Output('agent-init-div', 'style'),
                       dash.dependencies.Output('agent-init-params', 'children')],
                      [dash.dependencies.Input('add-modules-dropdown', 'value')],
                      )
        def add_agent_init_params(add_dropdown_val):
            # selected a class from the dropdown
            agentNetwork = app.dashboard_ctrl.agentNetwork
            if add_dropdown_val is not None:
                selected_agent_class = app.dashboard_ctrl.get_agentTypes()[add_dropdown_val]
                if hasattr(selected_agent_class,'parameter_choices'):
                    init_param_components =  [get_param_dash_component(key,val) for key,val in selected_agent_class.parameter_choices.items()]
                else:
                    init_param_components = []

                unique_agent_name =agentNetwork.generate_module_name_byType(selected_agent_class)

                return [unique_agent_name, {"display":"block"}, init_param_components]
            else:
                return ["", {"display":"none"},[]]

        # Add agent button click
        @app.callback(dash.dependencies.Output('add-module-button', 'children'),
                      [dash.dependencies.Input('add-module-button', 'n_clicks')],
                      [dash.dependencies.State('add-modules-dropdown', 'value'),
                       dash.dependencies.State('agent-init-name', 'value'),
                       dash.dependencies.State('agent-init-params', 'children'),
                       ]
                      )
        def add_agent_button_click(n_clicks, add_dropdown_val, init_name_input, init_params_div):
            # for add agent button click
            if n_clicks is not None:
                agentTypes = app.dashboard_ctrl.get_agentTypes()
                init_params_kwargs = extract_param_dropdown(init_params_div)
                new_agent = app.dashboard_ctrl.agentNetwork.add_agent(name=init_name_input, agentType=agentTypes[add_dropdown_val], **init_params_kwargs)
                raise_toast("Spawned new agent : %s !" % init_name_input)
            raise PreventUpdate

        # Add agent button click
        @app.callback(dash.dependencies.Output('remove-module-button', 'children'),
                      [dash.dependencies.Input('remove-module-button', 'n_clicks')],
                      [dash.dependencies.State('selected-node', 'children')]
                      )
        def remove_agent_button_click(n_clicks, current_agent_id):
            # for add agent button click
            if n_clicks is not None and current_agent_id != "Not selected":
                app.dashboard_ctrl.agentNetwork.remove_agent(current_agent_id)
                raise_toast("Removed agent : %s !" % current_agent_id)
            raise PreventUpdate

        # Add coalition button click
        @app.callback([dash.dependencies.Output('coalition-name', 'value')],
                      [dash.dependencies.Input('add-coalition-button', 'n_clicks')],
                      [dash.dependencies.State('coalition-name', 'value'),
                       ]
                      )
        def add_coalition_button_click(n_clicks, coalition_name):
            # for add agent button click
            if n_clicks is not None:
                if coalition_name != "":
                    new_coalition = app.dashboard_ctrl.agentNetwork.add_coalition(name=coalition_name)
                    raise_toast("Created new coalition : %s !" % coalition_name)
            return [""]


        @app.callback(dash.dependencies.Output('connect_placeholder', 'children'),
                      [dash.dependencies.Input('connect-module-button', 'n_clicks')],
                      [dash.dependencies.State('connect-modules-dropdown', 'value'),
                       dash.dependencies.State('selected-node', 'children')])
        def bind_module_click(n_clicks_connect, dropdown_value, current_agent_id):
            if n_clicks_connect is not None and current_agent_id != "Not selected":
                # get nameserver
                agentNetwork = app.dashboard_ctrl.agentNetwork
                target_agent = agentNetwork.get_agent(dropdown_value)
                # for connect module button click
                # connect agents
                if current_agent_id in agentNetwork.agents():
                    if current_agent_id != dropdown_value:
                        agentNetwork.bind_agents(agentNetwork.get_agent(current_agent_id),
                                                 target_agent)
                        raise_toast("Connected %s to %s !" % (current_agent_id, target_agent.name))
                # otherwise, selected is a coalition
                # add it into coalition
                else:
                    agentNetwork.add_coalition_agent(name=current_agent_id, agents=[target_agent])
                    raise_toast("%s joined %s coalition !" % (target_agent.name, current_agent_id))

            raise PreventUpdate

        @app.callback(dash.dependencies.Output('disconnect_placeholder', 'children'),
                      [dash.dependencies.Input('disconnect-module-button', 'n_clicks')],
                      [dash.dependencies.State('connect-modules-dropdown', 'value'),
                       dash.dependencies.State('selected-node', 'children')])
        def unbind_module_click(n_clicks_disconnect, dropdown_value, current_agent_id):
            if (n_clicks_disconnect is not None):
                # get nameserver
                agentNetwork = app.dashboard_ctrl.agentNetwork
                target_agent = agentNetwork.get_agent(dropdown_value)

                # for connect module button click
                # disconnect agents
                if current_agent_id in agentNetwork.agents():
                    if current_agent_id != dropdown_value:
                        agentNetwork.unbind_agents(agentNetwork.get_agent(current_agent_id),
                                                 target_agent)
                        raise_toast("Disconnected %s from %s !" % (current_agent_id, target_agent.name))
                # otherwise, selected is a coalition
                # remove it from coalition
                else:
                    agentNetwork.remove_coalition_agent(coalition_name=current_agent_id, agent_name=target_agent.name)
                    raise_toast("%s removed from %s coalition !" % (target_agent.name, current_agent_id))
            raise PreventUpdate

        @app.callback([dash.dependencies.Output('selected-node', 'children'),
                       dash.dependencies.Output('input-names', 'children'),
                       dash.dependencies.Output('output-names', 'children'),
                       dash.dependencies.Output('agent-parameters', 'children'),
                       ],
                      [dash.dependencies.Input('agents-network', 'tapNodeData')])
        def displayTapNodeData(data):
            input_names = []
            output_names = []
            agent_parameters_div = []

            if data is not None:
                current_agent_id = data['id']
                # get nameserver
                agentNetwork = app.dashboard_ctrl.agentNetwork

                # selected agent
                if current_agent_id in agentNetwork.agents():
                    app.dashboard_ctrl.current_selected_agent = agentNetwork.get_agent(current_agent_id)
                    input_names = list(app.dashboard_ctrl.current_selected_agent.get_attr('Inputs').keys())
                    output_names = list(app.dashboard_ctrl.current_selected_agent.get_attr('Outputs').keys())
                    agent_parameters = app.dashboard_ctrl.current_selected_agent.get_all_attr()

                    # formatting
                    input_names = ["Inputs: "] + [input_name + ", " for input_name in input_names]
                    output_names = ["Outputs: "] + [output_name + ", " for output_name in output_names]
                    agent_parameters_texts = [LayoutHelper.visualise_agent_parameters(k, v) for k, v in
                                              agent_parameters.items()]
                    agent_parameters_div = [html.H6("Parameters: ")] + agent_parameters_texts

                    return [current_agent_id,  input_names, output_names, agent_parameters_div]

                # selected coalition
                else:
                    coalition = agentNetwork.get_coalition(current_agent_id)
                    agent_members = [html.H6("Agents: ")] + [str(coalition.agent_names())]

                    return [current_agent_id,[], [], agent_members]

            else:
                return ["Not selected", input_names, output_names, agent_parameters_div]

        # define maximum number of monitors graph
        output_figures = [dash.dependencies.Output('monitors-graph-' + str(i), 'figure') for i in
                          range(app.num_monitors)]
        output_styles = [dash.dependencies.Output('monitors-graph-' + str(i), 'style') for i in range(app.num_monitors)]
        outputs = output_figures + output_styles

        @app.callback(outputs,
                      [dash.dependencies.Input('interval-update-monitor-graph', 'n_intervals')])
        def plot_monitor_memory(n_interval):
            # get nameserver
            agentNetwork = app.dashboard_ctrl.agentNetwork

            # check if agent network is running and first_time running
            # if it isn't, abort updating graphs
            if agentNetwork.get_mode() != "Running" and agentNetwork.get_mode() != "Reset" and n_interval > 0:
                raise PreventUpdate

            agent_names = agentNetwork.agents(filter_agent='Monitor')  # get all agent names
            app.num_monitor = len(agent_names)
            monitor_graphs = [{'data': []} for i in range(app.num_monitors)]
            style_graphs = [{'opacity': 0, 'width': 10, 'height': 10} for i in range(app.num_monitors)]

            for monitor_id, monitor_agent in enumerate(agent_names):
                monitor_buffer = agentNetwork.get_agent(monitor_agent).get_attr('buffer').buffer
                custom_plot_function = agentNetwork.get_agent(monitor_agent).get_attr('custom_plot_function')
                data = []
                for sender_agent, buffered_data in monitor_buffer.items():
                    # if custom plot function is not provided, resolve to default plotting
                    if custom_plot_function is None:
                        traces = create_monitor_graph(buffered_data, sender_agent)
                    # otherwise call custom plot function and load up custom plot parameters
                    else:
                        custom_plot_parameters = agentNetwork.get_agent(monitor_agent).get_attr(
                            'custom_plot_parameters')
                        # Handle iterable of traces.
                        traces = custom_plot_function(
                            buffered_data,
                            sender_agent,
                            **custom_plot_parameters
                        )

                    if (
                            isinstance(traces, tuple)
                            or isinstance(traces, list)
                            or isinstance(traces, set)
                    ):
                        for trace in traces:
                            data.append(trace)
                    else:
                        data.append(traces)

                if len(data) > 5:
                    y_title_offset = 0.1
                else:
                    y_title_offset = -0.1

                # Check if any metadata is present that can be used to generate axis
                # labels or otherwise use default labels.
                if (
                    len(monitor_buffer) > 0
                    and isinstance(buffered_data, dict)
                    and "metadata" in buffered_data.keys()
                ):
                    # The metadata currently is always a list in the
                    # beginning containing at least one element.
                    desc = buffered_data["metadata"][0]

                    # We now expect metadata to be of type
                    # time-series-metadata.scheme.MetaData. We try to access the
                    # object correspondingly and throw a meaningful error message in
                    # case something goes wrong.
                    try:
                        t_name, t_unit = desc.time.values()
                        v_name, v_unit = desc.get_quantity().values()
                    except TypeError:
                        raise TypeError(
                            f"The Dashboard tried to access an agents metadata but an"
                            f"error occurred. Metadata is " f"of type {type(desc)} "
                            f"but is expected to be of type ""{type(MetaData)}. " 
                            f"Its value is: \n\n{desc}"
                        )

                    # After successfully extracting the metadata itself, we concatenate
                    # the important parts to get the labels.
                    x_label = f"{t_name} [{t_unit}]"
                    y_label = f"{v_name} [{v_unit}]"
                else:
                    # If no metadata is available we set reasonable defaults. Since
                    # we could deal with any data in the time as well as in the
                    # frequency domain, we keep it fairly generic.
                    warnings.warn(
                        f"The Dashboard shows a plot for monitor agent '"
                        f"{monitor_agent}' without any axes labels specified. The "
                        f"labels will be represented by generic place holders. Check "
                        f"out tutorial 4 to find out how to specify custom labels."
                    )
                    x_label = "X"
                    y_label = "Y"

                monitor_graph = {
                    'data': data,
                    'layout': {
                        'title': {
                            'text': monitor_agent,
                            'y': y_title_offset,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'bottom'
                        },
                        'xaxis': {'title': {'text': x_label}},
                        'yaxis': {'title': {'text': y_label}},
                        'uirevision': app.num_monitor,
                        'showlegend': True,
                        'legend': dict(xanchor='auto', yanchor='bottom', x=1, y=1, orientation="h"),
                        # 'margin':dict(t=150)
                    },
                }

                monitor_graphs[monitor_id] = monitor_graph
                # style_graphs[monitor_id]= {'opacity':1.0, 'width':'100%','height':'100%'}
                style_graphs[monitor_id] = {'opacity': 1.0, 'height': 'auto'}
            # monitor_graphs = monitor_graphs+ [{'displayModeBar': False, 'editable': False, 'scrollZoom':False}]
            return monitor_graphs + style_graphs

        def _handle_matplotlib_figure(input_data, from_agent_name: str, mode="image"):
            """
            Internal function. Checks the mode of matplotlib.figure.Fig to be plotted
            Either it is a base64 str image, or a plotly graph

            This is used in plotting the received matplotlib figures in the MonitorAgent's plot memory.
            """
            if mode == "plotly":
                new_graph = dcc.Graph(figure=input_data)
            elif mode == "image":
                new_graph = html.Img(src=input_data, title=from_agent_name)
            elif mode == "mpld3":
                new_input_data = str(input_data).replace("'", '"')
                new_input_data = new_input_data.replace("(", "[")
                new_input_data = new_input_data.replace(")", "]")
                new_input_data = new_input_data.replace("None", "null")
                new_input_data = new_input_data.replace("False", "false")
                new_input_data = new_input_data.replace("True", "true")
                fig_json = html.P(new_input_data, style={'display': 'none'})
                new_graph = html.Div(id="d3_" + from_agent_name, children=fig_json)

            return new_graph

        # load Monitors data and draw - all at once
        @app.callback([dash.dependencies.Output('matplotlib-division', 'children')],
                      [dash.dependencies.Input('interval-update-monitor-graph', 'n_intervals')])
        def plot_monitor_graphs(n_interval):
            # get nameserver
            agentNetwork = app.dashboard_ctrl.agentNetwork

            # check if agent network is running and first_time running
            # if it isn't, abort updating graphs
            # if agentNetwork._get_mode() != "Running" and n_interval > 0:
            if agentNetwork.get_mode() != "Running" and n_interval > 0:
                raise PreventUpdate

            agent_name_filter = "Monitor"
            agent_names = agentNetwork.agents(filter_agent=agent_name_filter)  # get all agent names
              # all agents with Monitor in its name will be selected
            plots_data = {}  # storage for all monitor agent's memory

            # load data from all Monitor agent's memory
            for agent_name in agent_names:
                monitor_agent = agentNetwork.get_agent(agent_name)
                plots = monitor_agent.get_attr('plots')
                plots_data.update({agent_name: plots})

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
                html_div_monitor = []
                html_div_monitor.append(html.H5(agent_name, style={"text-align": "center"}))
                # create a new graph for every agent
                for from_agent_name in plot_data:
                    # get the graph relevant to 'monitor_agent_input'
                    graph = plot_data[from_agent_name]
                    print(graph)
                    # handle list of graphs
                    if (isinstance(graph["fig"], tuple) or isinstance(graph["fig"], list) or isinstance(graph["fig"],
                                                                                                        set)):
                        for graph_id, graph_ in enumerate(graph["fig"]):
                            new_graph = _handle_matplotlib_figure(graph_, from_agent_name + str(graph_id),
                                                                  graph["mode"])
                            html_div_monitor.append(new_graph)
                    else:
                        new_graph = _handle_matplotlib_figure(graph["fig"], from_agent_name, graph["mode"])
                        html_div_monitor.append(new_graph)

                # only add the graph if there is some plots in the Monitor Agent
                if len(html_div_monitor) > 1:
                    all_graphs.append(html.Div(className="card", children=html_div_monitor))

            # set dimensions of each monitor agent's graph
            return [all_graphs]

        @app.callback(
            dash.dependencies.Output('toast-js-script', 'run'),
            [dash.dependencies.Input('interval-update-toast', 'n_intervals')])
        def generate_toast_msg(n_intervals):
            # if there are messages in toast contents to be displayed
            if hasattr(app, "toast_contents") and len(app.toast_contents) > 0:
                # pop toast contents
                return "M.toast({html: '%s'})" % app.toast_contents.pop(0)

            else:
                return ""

        app.clientside_callback(
            ClientsideFunction(
                namespace='clientside',
                function_name='render_mpld3_each'
            ),
            dash.dependencies.Output('mpld3_placeholder', 'children'),
            [dash.dependencies.Input('matplotlib-division', 'children')]
        )

        app.toast_contents = []

        def raise_toast(message, app=app):
            app.toast_contents.append(message)

        return app
