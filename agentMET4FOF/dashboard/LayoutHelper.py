from typing import Dict, Union

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dash_table, dcc, html
from plotly.graph_objs import Scatter


# return icon and text for button
def html_icon(icon="play_circle_filled", text="Start", alignment="left"):
    return [html.I(icon, className="material-icons " + alignment), text]


def html_button(icon="play_circle_filled", text="Start", id=" ", style={}):
    new_style = {"margin-top": "10px"}
    new_style.update(style)
    return html.Button(
        children=html_icon(icon, text), id=id, className="btn", style=new_style
    )


def create_nodes_cytoscape(agent_graph):
    pos = nx.fruchterman_reingold_layout(agent_graph)
    new_elements = [
        {
            "data": {"id": k, "label": k},
            "position": {"x": pos[k][0], "y": pos[k][1]},
            "classes": agent_graph.nodes[k]["stylesheet"],
        }
        for k in agent_graph.nodes()
    ]

    return new_elements


def create_edges_cytoscape(edges, hide_default_edge=True):
    """
    Converts data of network edges into a form accepted by cytoscape network generation in the dashboard.

    Parameters
    ----------
    edges : list
        List of edges where index 0 and 1 are names of sources and targets respectively. Index 2 is a dictionary with key of 'channel'.

    hide_default_edge : boolean
        Set to hide 'default' channel display or not.
    """

    new_elements = []
    for edge in edges:
        if (
            hide_default_edge
            and isinstance(edge[2]["channel"], str)
            and edge[2]["channel"] == "default"
        ):
            new_elements += [{"data": {"source": edge[0], "target": edge[1]}}]
        else:
            new_elements += [
                {
                    "data": {
                        "source": edge[0],
                        "target": edge[1],
                        "channel": edge[2]["channel"],
                    }
                }
            ]
    return new_elements


def create_monitor_graph(
    data: Union[Dict, np.ndarray], sender_agent: str = "Monitor Agent"
) -> Scatter:
    """
    Parameters
    ----------
    data : dict or np.ndarray
        The data saved in the MonitorAgent's buffer, for each Inputs (Agents) it is
        connected to.
    sender_agent : str
        Name of the sender agent

    Returns
    -------
    Scatter
        Plotly scatter plot of the data
    """
    if isinstance(data, dict):
        if "time" not in data.keys():
            trace = [
                go.Scatter(
                    x=np.arange(len(data[key])),
                    y=data[key],
                    mode="lines",
                    name=f"{sender_agent}:{key}",
                )
                for key in data.keys()
            ]
        else:
            trace = [
                go.Scatter(
                    x=data["time"],
                    y=data[key],
                    mode="lines",
                    name=f"{sender_agent}:{key}",
                )
                for key in data.keys()
                if key != "time"
            ]
    else:
        y = data
        x = np.arange(len(y))
        trace = go.Scatter(x=x, y=y, mode="lines", name=sender_agent)
    return trace


def create_params_table(table_name="", data={}, columns=None, **kwargs):
    style_table = {"overflowX": "scroll"}
    style_cell = {
        "minWidth": "0px",
        "maxWidth": "180px",
        "whiteSpace": "normal",
        "font_size": "14px",
    }
    css = [
        {
            "selector": ".dash-cell div.dash-cell-value",
            "rule": "display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;",
        }
    ]

    # if data is null, return early
    if len(data) == 0:
        output_info_table = dash_table.DataTable(
            id=table_name,
            style_table=style_table,
            style_cell=style_cell,
            css=css,
        )
        return output_info_table

    # convert dict into pandas
    elif type(data) == dict and len(data) > 0:
        data_pd = pd.DataFrame.from_dict(data)
        data = data_pd.reset_index().astype(str)

    if columns is None:
        columns = [{"name": i, "id": i} for i in data.columns]
    else:
        columns = [{"name": i, "id": i} for i in columns]

    data = data.to_dict("records")

    output_info_table = dash_table.DataTable(
        id=table_name,
        columns=columns,
        data=data,
        style_table=style_table,
        style_cell=style_cell,
        css=css,
        **kwargs,
    )
    return output_info_table


def visualise_agent_parameters(k, v):
    if k == "output_channels_info":
        output_info_table = create_params_table("agent-parameters-table", v)
        return html.Div([html.H6(k), output_info_table])
    else:
        return html.H6(k + ": " + str(v))


def get_param_dash_component(param_key, param_set):
    """
    Converts param_key:iterable (param_set) into a list of dash dropdowns
    """
    if isinstance(param_set, set) or isinstance(param_set, list):
        dropdown_options = [{"label": param, "value": param} for param in param_set]
        return dcc.Dropdown(options=dropdown_options, placeholder=param_key)
    else:
        return []


def extract_param_dropdown(params_div):
    """
    Extracts parameters from the init_param dropdown list.
    These extracted parameters will be passed to the agent's initialisation in the add agent button.
    """
    init_params = {}
    for div in params_div:
        if isinstance(div, dict):
            if "value" in div["props"].keys():
                init_params.update({div["props"]["placeholder"]: div["props"]["value"]})
    return init_params
