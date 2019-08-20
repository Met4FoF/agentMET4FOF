import dash_html_components as html
import networkx as nx
import numpy as np
import plotly.graph_objs as go


#return icon and text for button
def html_icon(icon="play_circle_filled" ,text="Start", alignment="left"):
    return [html.I(icon, className="material-icons "+alignment),text]

def html_button(icon="play_circle_filled", text="Start",id=" ", style ={}):
    new_style = {'margin-top': '10px'}
    new_style.update(style)
    return html.Button(children=html_icon(icon,text), id=id, className="btn waves-light", style=new_style)

def create_nodes_cytoscape(agent_graph):
    pos = nx.fruchterman_reingold_layout(agent_graph)
    new_elements = [{'data': {'id': k, 'label': k}, 'position': {'x': pos[k][0], 'y': pos[k][1]}} for k in agent_graph.nodes()]

    return new_elements

def create_edges_cytoscape(edges):
    new_elements =[]
    for edge in edges:
        new_elements += [{'data': {'source': edge[0], 'target': edge[1]}}]
    return new_elements

def create_monitor_graph(data):
    y = data
    x = np.arange(len(y))
    trace = go.Scatter(x=x, y=y,mode="lines", name='Monitor Agent')
    return trace
