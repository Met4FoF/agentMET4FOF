
# coding: utf-8

# ## Agent Dashboard User Interface
# 
# 1. Run this code after the Agents are setup and running 
# 2. View the web visualization at port 8054 using Internet Browser

# In[1]:


portNumber = 8054


# In[2]:


import osbrain
from osbrain.agent import run_agent
from osbrain import NSProxy

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_cytoscape as cyto
import dash_daq as daq
import plotly.graph_objs as go
import networkx as nx
import numpy as np

import pickle
external_css = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "https://fonts.googleapis.com/css?family=Raleway:400,400i,700,700i",
                "https://fonts.googleapis.com/css?family=Product+Sans:400,400i,700,700i"]
app = dash.Dash(__name__, external_stylesheets=external_css)

#===============APP LAYOUT=========
agent_names =['aggregator_1', 'sensor_0', 'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'predictor_0', 'decisionMaker_0']

G = nx.Graph()
G.add_nodes_from(agent_names)
myEdges = []

for agent_x in agent_names:
    for agent_y in agent_names:
        include=False
        if 'sensor' in agent_x and 'aggregator' in agent_y:
            include = True
        elif 'aggregator' in agent_x and 'predictor' in agent_y:
            include = True
        elif 'predictor' in agent_x and 'decisionMaker' in agent_y:
            include = True
        if include:
            new_edge = (agent_x, agent_y)
            myEdges.append(new_edge)

G.add_edges_from(myEdges)
pos=nx.fruchterman_reingold_layout(G)

nodes_ct = [{'data': {'id': k, 'label': k}, 'position': {'x': pos[k][0], 'y': pos[k][1]},'classes': k.split('_')[0]} for k in agent_names]

edges_ct = [{'data': {'source': k[0], 'target': k[1]}} for k in myEdges]
elements = nodes_ct+edges_ct

tab_div_style ={
            "padding": "2",
            "marginLeft": "5",
            "marginRight": "5",
            "backgroundColor":"white",
            "border": "1px solid #C8D4E3",
            "borderRadius": "3px"
        }
tab_title = { 'textAlign': 'center',}

output_labels = [{0: "Optimal", 1: "Reduced", 2: "Nearly Fail"},
 {0: "Optimal", 1: "Small lag", 2: "Severe lag", 3: "Nearly Fail"},
 {0: "No Leakage", 1: "Weak Leakage", 2: "Severe Leakage"},
 {0: "Optimal", 1: "Slightly Reduced", 2: "Severely Reduced", 3: "Nearly Fail"},
 {0: "Stable", 1: "Unstable"}]

output_category = ["Cooler Condition", "Valve Condition", "Internal Pump", "Accumulator", "Stable Flag"]

def getConditionIndicator(condition_text="Optimal",certain=True):
    color = "#00cc96" if certain else "#ff0000"
    label = "Certain" if certain else "Uncertain"
    return [
        html.H5(condition_text, style=tab_title),
        daq.Indicator(
            value=True,
            color=color,
            label=label,
            style=tab_title
        )]

app.layout = html.Div(children=[
html.Div([
        html.H3("Multi Agents for Machine Learning under Uncertainty Testbed",style={
            'textAlign': 'center',

        }),
        ]),
    html.Div([
        html.Div([
            html.H6("Cooler Condition",style= { 'textAlign': 'center',"border": "1px solid #C8D4E3"}),
            html.Div(children=getConditionIndicator("Optimal",certain=True),id='cooler-indicator')
        ],className="two columns",style=tab_div_style),
        html.Div([
            html.H6("Valve Condition", style={'textAlign': 'center', "border": "1px solid #C8D4E3"}),
            html.Div(children=getConditionIndicator("Optimal", certain=True),id='valve-indicator')
        ], className="two columns", style=tab_div_style),
        html.Div([
            html.H6("Internal Pump", style={'textAlign': 'center', "border": "1px solid #C8D4E3"}),
            html.Div(children=getConditionIndicator("Optimal", certain=True),id='pump-indicator')
        ], className="two columns", style=tab_div_style),
        html.Div([
            html.H6("Accumulator", style={'textAlign': 'center', "border": "1px solid #C8D4E3"}),
            html.Div(children=getConditionIndicator("Optimal", certain=True),id='accumulator-indicator')
        ], className="two columns", style=tab_div_style),
        html.Div([
            html.H6("Stable Flag", style={'textAlign': 'center', "border": "1px solid #C8D4E3"}),
            html.Div(children=getConditionIndicator("Optimal", certain=True),id='stability-indicator')
        ], className="two columns", style=tab_div_style),
    ],className="row"),

    html.Div([
        html.Div([
            html.H5('Agent Network Dashboard', style=tab_title
        ),
        cyto.Cytoscape(
            id='agents-network',
            layout={'name': 'circle'},
            style={'width': '100%', 'height': '400px'},
            elements=elements,
            stylesheet= [
                {
                    'selector': 'node',
                    'style': {
                        'label': 'data(id)'
                    }
                },
                {
                'selector': '.sensor',
                'style': {
                    'background-color': 'green',
                    'line-color': 'black'
                }
            },
                {
                'selector': '.aggregator',
                'style': {
                    'background-color': 'blue',
                    'line-color': 'black'
                }
            },
                {
                    'selector': '.predictor',
                    'style': {
                        'background-color': 'red',
                        'line-color': 'black'
                    }
                },
                {
                    'selector': '.decisionMaker',
                    'style': {
                        'background-color': 'yellow',
                        'line-color': 'black'
                    }
                },
            ]
        ),], className="six columns",style=tab_div_style),
        html.Div([
            html.H5(
                children='Select Predictor Agent',
                style={
                    'textAlign': 'left',
                }
            ),
            html.Div(
                id='predictor-dropdown-div',
                children=dcc.Dropdown(
                    id='predictor-dropdown',
                    options=[],
                    value='predictor_0',
                    style={'width': 250},
                )),
            html.Div([
                html.H5("Prediction Graph",style=tab_title)
            ]),
            dcc.Graph(id='prediction-graph'),
        ], className='six columns ', style=tab_div_style)
],className="row"),

    html.Div([

        html.Div([
            html.Div([
                html.H5("Sensor Graph",style=tab_title)
            ]),
            html.H5(
                children='Select Sensor Agent',
                style={
                    'textAlign': 'left',
                }
            ),
            html.Div(
                id='sensor-dropdown-div',
                children=dcc.Dropdown(
                    id='sensor-dropdown',
                    options=[],
                    value='sensor_number_0',
                    style={'width': 250},
                )),
            dcc.Graph(id='sensor-graph'),
        ], className='six columns ',style=tab_div_style),
        html.Div([
            html.Div([
                html.H5("Uncertainty Graph",style=tab_title)
            ]),
            dcc.Graph(id='uncertainty-graph'),
        ], className='six columns ',style=tab_div_style),
    ], className='row'),

    dcc.Interval(
        id='interval-component',
        interval=3 * 1000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Interval(
        id='interval-component-network-graph',
        interval=1000 * 1000,  # in milliseconds
        n_intervals=0
    )


],style={
            "padding": "8",
            "marginLeft": "45",
            "marginRight": "45",
            "backgroundColor":"white",
            "border": "1px solid #C8D4E3",
            "borderRadius": "3px"
        })

@app.callback(dash.dependencies.Output('sensor-dropdown', 'value'),
              [dash.dependencies.Input('agents-network', 'tapNodeData')])
def displayTapNodeData(data):
    if 'sensor' in data['label'] and 'sensor_network' not in data['label']:
        return data['label']

@app.callback([dash.dependencies.Output('agents-network', 'elements'),dash.dependencies.Output('sensor-dropdown-div', 'children'),dash.dependencies.Output('predictor-dropdown-div', 'children')],
              [dash.dependencies.Input('interval-component-network-graph', 'n_intervals')])
def update_network_graph(n):
    ns_temp = NSProxy(nsaddr='127.0.0.1:14065')
    agent_names = ns_temp.agents()
    print(agent_names)
    G = nx.Graph()
    G.add_nodes_from(agent_names)
    myEdges = []

    for agent_x in agent_names:
        for agent_y in agent_names:
            include = False
            if 'sensor' in agent_x and 'aggregator' in agent_y and 'sensor_network' not in agent_x:
                include = True
            elif 'aggregator' in agent_x and 'predictor' in agent_y:
                include = True
            elif 'predictor' in agent_x and 'decisionMaker' in agent_y:
                include = True
            if include:
                new_edge = (agent_x, agent_y)
                myEdges.append(new_edge)

    G.add_edges_from(myEdges)
    pos = nx.fruchterman_reingold_layout(G)

    nodes_ct = [{'data': {'id': k, 'label': k}, 'position': {'x': pos[k][0], 'y': pos[k][1]},'classes': k.split('_')[0]} for k in agent_names]
    edges_ct = [{'data': {'source': k[0], 'target': k[1]}} for k in myEdges]
    elements = nodes_ct + edges_ct

    sensor_options = [{'label': name, 'value': name} for name in agent_names if 'sensor' in name and 'sensor_network' not in name]
    predictor_options = [{'label': name, 'value': name} for name in agent_names if 'predictor' in name ]

    sensor_dropdown_component = dcc.Dropdown(
        id='sensor-dropdown',
        options=sensor_options,
        value=sensor_options[0]['value'],
        style={'width': 250},
    )
    predictor_dropdown_component = dcc.Dropdown(
        id='predictor-dropdown',
        options=predictor_options,
        value=predictor_options[0]['value'],
        style={'width': 250},
    )
    print("HELLO")
    return [elements, sensor_dropdown_component,predictor_dropdown_component]

@app.callback(dash.dependencies.Output('sensor-graph', 'figure'),
              [dash.dependencies.Input('interval-component', 'n_intervals'),dash.dependencies.Input('sensor-dropdown', 'value')])
def update_sensor_graph(n,chosen_sensor_name):
    ns_temp = NSProxy(nsaddr='127.0.0.1:14065')
    sensor_type = ns_temp.proxy(chosen_sensor_name).get_attr('type')
    sensor_unit = ns_temp.proxy(chosen_sensor_name).get_attr('unit_v')

    final_data = ns_temp.proxy(chosen_sensor_name).get_attr('current_data')

    final_data = np.array(final_data)

    y_data = final_data
    N_sequence = y_data.shape[0]
    x_data = np.linspace(0, N_sequence - 1, N_sequence)

    traces_sensor = go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        name='lines'
    )
    layout = {'title': 'Sensor '+sensor_type+" #"+chosen_sensor_name.split('_')[-1],
              'xaxis': {'title': 'Time (s)'},
              'yaxis': {'title': sensor_type+" ("+sensor_unit+")"},
              }

    return {
        'data': [traces_sensor],
        'layout': layout
    }

predictions = []
uncertainties = []
probabilities_accurate = []

def getTimeSeriesGraph(y_data,title='Prediction Certainty vs Time',xaxis='Time (s)',yaxis='Certainty (%) '):
    N_sequence = y_data.shape[0]
    x_data = np.linspace(0, N_sequence - 1, N_sequence)

    traces_sensor = go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        name='lines'
    )
    layout = {'title': title,
              'xaxis': {'title': xaxis},
              'yaxis': {'title': yaxis},
              }

    return {
        'data': [traces_sensor],
        'layout': layout
    }

@app.callback([dash.dependencies.Output('prediction-graph', 'figure'), dash.dependencies.Output('uncertainty-graph', 'figure'), dash.dependencies.Output('cooler-indicator', 'children'), dash.dependencies.Output('valve-indicator', 'children'), dash.dependencies.Output('pump-indicator', 'children'), dash.dependencies.Output('accumulator-indicator', 'children'), dash.dependencies.Output('stability-indicator', 'children')],
              [dash.dependencies.Input('interval-component', 'n_intervals'),dash.dependencies.Input('predictor-dropdown', 'value')])
def update_prediction_graph(n,selected_predictor):
    ns_temp = NSProxy(nsaddr='127.0.0.1:14065')
    overall_new_data = ns_temp.proxy('decisionMaker_0').get_attr('current_inference')
    column_numbers = [x.split('_')[-1] for x in overall_new_data.columns]
    overall_new_data.columns = column_numbers
    overall_new_data.sort_index(axis=1, inplace=True)
    #condition_indicators = [getConditionIndicator(output_labels[int(predictor['label'].split('_')[-1])][overall_new_data[predictor['label']].pred],certain=overall_new_data[predictor['label']].unc_state) for id_,predictor in enumerate(predictors) ]
    print(overall_new_data)
    print(overall_new_data.loc['pred'])


    new_prediction = overall_new_data.loc['pred']
    new_uncertainty = overall_new_data.loc['unc']
    new_uncertainty_state = overall_new_data.loc['unc_state']

    condition_indicators = [getConditionIndicator(output_labels[_id][new_prediction[_id]],certain=unc_state ) for _id,unc_state in enumerate(new_uncertainty_state)]

    predictions.append(new_prediction)
    uncertainties.append(new_uncertainty)
    selected_predictor_id=int(selected_predictor.split("_")[-1])
    # dim 0 = samples, dim 1 = predictor
    y_data_pred = np.array(predictions)[:,selected_predictor_id]
    y_data_unc = np.array(uncertainties)[:, selected_predictor_id]

    #pred
    prediction_graph = getTimeSeriesGraph(y_data_pred,'Model Prediction ('+output_category[selected_predictor_id]+') vs Time','Time (s)','Prediction ')
    uncertainties_graph = getTimeSeriesGraph(y_data_unc,'Prediction Certainty vs Time','Time (s)','Certainty (%) ')


    return [prediction_graph,uncertainties_graph] +condition_indicators

if __name__ == '__main__':

    app.run_server(debug=False, port=portNumber)



