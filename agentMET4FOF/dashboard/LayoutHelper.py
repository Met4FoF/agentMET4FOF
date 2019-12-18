import dash_html_components as html
import dash_table

import networkx as nx
import numpy as np
import plotly.graph_objs as go
import pandas as pd


#return icon and text for button
def html_icon(icon="play_circle_filled" ,text="Start", alignment="left"):
    return [html.I(icon, className="material-icons "+alignment),text]

def html_button(icon="play_circle_filled", text="Start",id=" ", style ={}):
    new_style = {'margin-top': '10px'}
    new_style.update(style)
    return html.Button(children=html_icon(icon,text), id=id, className="btn", style=new_style)

def create_nodes_cytoscape(agent_graph):
    pos = nx.fruchterman_reingold_layout(agent_graph)
    new_elements = [{'data': {'id': k, 'label': k}, 'position': {'x': pos[k][0], 'y': pos[k][1]}} for k in agent_graph.nodes()]

    return new_elements

def create_edges_cytoscape(edges):
    new_elements =[]
    for edge in edges:
        new_elements += [{'data': {'source': edge[0], 'target': edge[1]}}]
    return new_elements

def create_monitor_graph(data,sender_agent = 'Monitor Agent'):
    y = data
    x = np.arange(len(y))
    trace = go.Scatter(x=x, y=y,mode="lines", name=sender_agent)
    return trace

def create_params_table(table_name="",data={}, columns=None, **kwargs):
    style_table = {'overflowX': 'scroll'}
    style_cell = {
            'minWidth': '0px', 'maxWidth': '180px',
            'whiteSpace': 'normal',
            'font_size': '14px',
        }
    css = [{
            'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
        }]

    #if data is null, return early
    if len(data) == 0:
        output_info_table = dash_table.DataTable(
            id=table_name,
            style_table=style_table,
            style_cell=style_cell,
            css=css,
        )
        return output_info_table

    #convert dict into pandas
    elif type(data) == dict and len(data)>0:
        data_pd = pd.DataFrame.from_dict(data)
        data = data_pd.reset_index().astype(str)

    if columns is None:
        columns= [{"name": i, "id": i} for i in data.columns]
    else:
        columns=[{"name": i, "id": i} for i in columns]

    data= data.to_dict('records')

    output_info_table = dash_table.DataTable(
        id=table_name,
        columns=columns,
        data=data,
        style_table=style_table,
        style_cell=style_cell,
        css=css,
        **kwargs
    )
    return output_info_table

def visualise_agent_parameters(k,v):
    if k == "output_channels_info":
        output_info_table = create_params_table('agent-parameters-table',v)
        return html.Div([html.H6(k),output_info_table])
    else:
        return html.H6(k +": "+str(v))
