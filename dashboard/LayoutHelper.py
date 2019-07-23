import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import networkx as nx
import numpy as np

#return icon and text for button
def html_icon(icon="play_circle_filled" ,text="Start", alignment="left"):
    return [html.I(icon, className="material-icons "+alignment),text]

def html_button(icon="play_circle_filled", text="Start",id=" "):
    return html.Button(children=html_icon(icon,text), id=id, className="btn waves-light", style={'margin-top': '10px'})

def get_nodes(agent_network):
    agent_names = agent_network.agents()
    return agent_names

def get_edges(agent_network):
    edges = []
    agent_names = agent_network.agents()

    for agent_name in agent_names:
        temp_agent = agent_network.get_agent(agent_name)
        temp_output_connections = list(temp_agent.get_attr('Outputs').keys())
        for output_connection in temp_output_connections:
            edges += [(temp_agent.get_attr('name'), output_connection)]
    return edges

def create_nodes(agent_graph):
    pos = nx.fruchterman_reingold_layout(agent_graph)
    new_elements = [{'data': {'id': k, 'label': k}, 'position': {'x': pos[k][0], 'y': pos[k][1]}} for k in agent_graph.nodes()]

    return new_elements

def create_edges(edges):
    new_elements =[]
    for edge in edges:
        new_elements += [{'data': {'source': edge[0], 'target': edge[1]}}]
    return new_elements

def create_monitor_graph(data):
    y = data
    x = np.arange(len(y))
    trace = go.Scatter(x=x, y=y,mode="lines", name='Monitor Agent')
    return trace
