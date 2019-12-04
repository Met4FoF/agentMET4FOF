import dash_html_components as html
import dash_core_components as dcc
import dash
import os
import time
import dash_table
from dash.exceptions import PreventUpdate
import agentMET4FOF.dashboard.LayoutHelper as LayoutHelper
from agentMET4FOF.dashboard.LayoutHelper import create_nodes_cytoscape, create_edges_cytoscape, create_monitor_graph
from datetime import datetime
import pandas as pd
from agentMET4FOF.develop.ML_Experiment import load_experiment
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from math import log10, floor

experiment_list = []
# experiment_base_path ="F:/PhD Research/Github/develop_ml_experiments_met4fof/agentMET4FOF/examples/ML_EXPERIMENTS/"
experiment_base_path=""
experiment_folder = "ML_EXP"
experiment_path = experiment_base_path+experiment_folder
# print(os.listdir(experiment_path))
# os.chdir(experiment_path)
for dir_ in os.listdir(experiment_base_path+experiment_folder):
    if "." not in dir_:
        date_mod = datetime.strptime(time.ctime(os.path.getmtime(experiment_base_path+experiment_folder+"/"+dir_)), "%a %b %d %H:%M:%S %Y")
        date_mod_string = date_mod.strftime("%d-%m-%Y, %H:%M")
        experiment_list.append({"Name":dir_,"__date_time":date_mod, "Date": date_mod_string})

df = pd.DataFrame(experiment_list).sort_values(by=["__date_time","Name"], ascending=False)

df = df[["Name","Date"]]

external_stylesheets= ['https://fonts.googleapis.com/icon?family=Material+Icons']
external_scripts = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js']

app = dash.Dash(__name__,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts
                )

ml_exp = load_experiment(ml_experiment_name="run_2")
pipeline_data = pd.DataFrame.from_dict(ml_exp.pipeline_details)

def get_ml_exp_layout():
                #body
    return html.Div(className="row",children=[

        #main panel
        html.Div(className="col s8", children=[
                html.Div(className="card", children=[
                   html.Div(className="card-content", children=[
                           html.Span(className="card-title", children=["Results"]),
                    ]),
                   html.Div(className="card-action", id="chains-div", children=[
                    LayoutHelper.create_params_table(table_name="chains-table",
                                                     data=df,
                                                        editable=True,
                                                        filter_action="native",
                                                        sort_action="native",
                                                        sort_mode="multi",
                                                        row_selectable="multi",
                                                        selected_rows=[],
                                                     ),
                    ])

                ]),
                html.Div(className="card", id="compare-graph-div", children=[])

        ]),

        #side panel
        html.Div(className="col s4", children=[
            html.Div(className="card blue lighten-4", children= [
                html.Div(className="card-content", children=[

                    html.Span(className="card-title",style={'margin-top': '20px'}, children=[
                        # html.H6(className="black-text", children="ML Experiments"),
                         "ML Experiments"
                    ]),

                    LayoutHelper.create_params_table(table_name="experiment-table",
                                                     data=df,
                                                        editable=True,
                                                        filter_action="native",
                                                        sort_action="native",
                                                        sort_mode="multi",
                                                        row_selectable="multi",
                                                        selected_rows=[],
                                                     ),
                    html.H6([""],id="experiment-placeholder-selected-rows")
                ])
            ]),

            html.Div(className="card blue lighten-4", children= [
                html.Div(className="card-content", children=[

                    html.Span(className="card-title",style={'margin-top': '20px'}, children=[
                         "Pipelines"
                    ]),
                    html.Div(id="pipeline-div",children=
                    LayoutHelper.create_params_table(table_name="pipeline-table",
                                                    data=df,
                                                    editable=True,
                                                    filter_action="native",
                                                    sort_action="native",
                                                    sort_mode="multi",
                                                    row_selectable="multi",
                                                    selected_rows=[],
                                                     )
                    )

                ])
            ]),


        ]),
    ])

app.layout = get_ml_exp_layout()

app.ml_experiments = []
app.aggregated_chain_results ={}

@app.callback(
    [Output('experiment-placeholder-selected-rows', "children"),
     # Output('pipeline-table', "columns"),
    Output('pipeline-div', "children"),
     Output('chains-div', "children"),
     ],
    [Input('experiment-table', "derived_virtual_data"),
     Input('experiment-table', "derived_virtual_selected_rows")])
def update_experiment_table(rows, derived_virtual_selected_rows):
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []
        raise PreventUpdate

    ml_experiments = []
    pipeline_details = []
    chain_results =[]

    #collect all selected experiments
    for selected_experiment in derived_virtual_selected_rows:
        ml_experiments.append(load_experiment(ml_experiment_name=rows[selected_experiment]['Name']))

    #extract pipeline details from every experiment
    if len(ml_experiments) != 0:
        for ml_experiment in ml_experiments:
            pipeline_details = pipeline_details+ml_experiment.pipeline_details
            chain_results = chain_results+ml_experiment.chain_results

        pipeline_details = pd.DataFrame(pipeline_details)
        pipeline_details_string = pipeline_details.applymap(str)

        #compute aggregated results
        chain_results = pd.DataFrame.from_dict(chain_results)
        aggregated_chain_results = chain_results[["chain","evaluation"]]

        mean_chain = aggregated_chain_results.groupby('chain').mean()["evaluation"]
        std_chain = aggregated_chain_results.groupby('chain').std()["evaluation"]


        aggregated_chain_results = pd.concat([mean_chain, std_chain], axis=1).applymap(round_sig)

        aggregated_chain_results= aggregated_chain_results.reset_index()
        aggregated_chain_results.columns = ["chain","Mean","Std"]

        #auto select all
        selected_rows_chains = np.arange(0,aggregated_chain_results.shape[0])
    else:
        #to be published in tables
        pipeline_details_string = {}
        chain_results = {}
        aggregated_chain_results = {}
        selected_rows_chains = [

        ]
    #save into persistent storage
    app.ml_experiments = ml_experiments
    app.pipeline_details = pipeline_details
    app.chain_results = chain_results
    app.aggregated_chain_results = aggregated_chain_results

    #create the pipeline table
    pipeline_table= LayoutHelper.create_params_table(table_name="pipeline-table",
                                                    data=pipeline_details_string,
                                                    editable=True,
                                                    filter_action="native",
                                                    sort_action="native",
                                                    sort_mode="multi",
                                                    row_selectable="multi",
                                                    selected_rows=[],
                                                     )


    #create chains table
    # chains_table= LayoutHelper.create_params_table(table_name="chains-table",
    #                                                 data=aggregated_chain_results,
    #                                                 editable=True,
    #                                                 filter_action="native",
    #                                                 sort_action="native",
    #                                                 sort_mode="multi",
    #                                                 row_selectable="multi",
    #                                                 selected_rows=selected_rows_chains,
    #                                                 style_data={
    #                                                     'whiteSpace': 'normal',
    #                                                     'height': 'auto'
    #                                                     },
    #                                                  )
    try:
        chains_table = dash_table.DataTable(data=aggregated_chain_results.to_dict('records'),
                                             columns=[{'id': c, 'name': c} for c in aggregated_chain_results.columns],
                                             style_data={
                                                    'whiteSpace': 'normal',
                                                    'height': 'auto',
                                                    'text-align':'left'
                                                },
                                             row_selectable="multi",
                                             selected_rows=selected_rows_chains,
                                             filter_action="native",
                                             sort_mode="multi",
                                             id="chains-table"
                                             )
    except:
        chains_table = " "
    return [str(rows), pipeline_table, chains_table]


def round_sig(x, sig=2):
    try:
        return round(x, sig-int(floor(log10(abs(x))))-1)
    except:
        return x

@app.callback(
    [Output('compare-graph-div', "children")],
    [Input('chains-table', "derived_virtual_data"),
     Input('chains-table', "derived_virtual_selected_rows")]
    )
def update_experiment_table(derived_virtual_data, derived_virtual_selected_rows):
    if derived_virtual_selected_rows is None or len(derived_virtual_selected_rows) == 0:
        derived_virtual_selected_rows = []
        return [""]

    aggregated_chain_results = app.aggregated_chain_results
    if len(aggregated_chain_results) == 0:
        return [""]

    aggregated_chain_results = aggregated_chain_results.sort_values(by=['Mean'],ascending=False)
    filter_chain = [derived_virtual_data[selected_chain]['chain'] for selected_chain in derived_virtual_selected_rows]
    aggregated_chain_results = aggregated_chain_results[aggregated_chain_results['chain'].isin(filter_chain)]

    final_graphs = [
        dcc.Graph(
            id=column + '--row-ids',
            figure={
                'data': [
                    {
                        'x': aggregated_chain_results['chain'],
                        'y': aggregated_chain_results[column],
                        'type': 'bar',
                        # 'marker': {'color': colors},
                    }
                ],
                'layout': {
                    'xaxis': {'automargin': True},
                    'yaxis': {
                        'automargin': True,
                        'title': {'text': column}
                    },
                    'height': 250,
                    'margin': {'t': 10, 'l': 10, 'r': 10},
                },
            },
        )
        # check if column exists - user may have deleted it
        # If `column.deletable=False`, then you don't
        # need to do this check.
        for column in ['Mean', 'Std'] if column in aggregated_chain_results
    ]

    return [final_graphs]


if __name__ == '__main__':
    app.run_server(debug=False, port=8051)
