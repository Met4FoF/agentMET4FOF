"""
Provides functions to obtain data from experiments folder (default is MLEXP)
To be rendered on the tab of the dashboard for visualizing and comparing ML experiments
"""

import dash_html_components as html
import dash_core_components as dcc
import os
import time
import dash_table
from dash.exceptions import PreventUpdate
from . import LayoutHelper
from datetime import datetime
import pandas as pd
from ..develop.ML_Experiment import load_experiment
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from math import log10, floor
import copy

def get_experiments_list():
    try:
        #get experiments list
        experiment_list = []
        experiment_base_path = ""
        experiment_folder = "ML_EXP"
        experiment_path = experiment_base_path+experiment_folder
        for dir_ in os.listdir(experiment_base_path+experiment_folder):
            if "." not in dir_:
                temp_exp = load_experiment(ml_experiment_name=dir_)

                experiment_meta_data = {"Name":temp_exp.name,"Date":temp_exp.run_date_string,
                                        "Datasets":str(temp_exp.datasets), "Mode":temp_exp.train_mode,"__date_time":temp_exp.run_date}

                experiment_list.append(experiment_meta_data)
        experiment_list = pd.DataFrame(experiment_list).sort_values(by=["__date_time","Name"], ascending=False)
        experiment_list = experiment_list.drop(columns=["__date_time"])
    except Exception as e:
        experiment_list = {}
    return experiment_list

def get_ml_exp_layout(experiments_df={}):
    #body
    return html.Div(className="row",children=[
        #main panel
        html.Div(className="col s8", children=[
                html.Div(className="card", children=[
                   html.Div(className="card-content", children=[
                           html.Span(className="card-title", children=["Results"]),
                                       html.Div(className="row", children = [
                                                    html.Div(className="col", children=[
                                                        LayoutHelper.html_button(icon="select_all",text="Select All", id="select-all-button")
                                                    ]),
                                                    html.Div(className="col", children=[
                                                        LayoutHelper.html_button(icon="tab_unselected",text="Unselect All", id="unselect-all-button")
                                                    ]),
                                                ])
                    ]),
                   html.Div(className="card-action", id="chains-div", children=[
                    LayoutHelper.create_params_table(table_name="chains-table",
                                                     data={},
                                                        editable=True,
                                                        filter_action="native",
                                                        sort_action="native",
                                                        sort_mode="multi",
                                                        row_selectable="multi",
                                                        selected_rows=[],
                                                     ),
                    ])

                ]),
                html.Div(className="card", id="compare-graph-div", children=[]),
                html.Div(id="placeholder-select",style={"opacity":0}),
        ]),

        #side panel
        html.Div(className="col s4", children=[
            html.Div(className="card blue lighten-4", children= [
                html.Div(className="card-content", children=[

                    html.Span(className="card-title",style={'margin-top': '20px'}, children=[
                         "ML Experiments"
                    ]),
                    LayoutHelper.create_params_table(table_name="experiment-table",
                                                     data=experiments_df,
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
                                                    data={},
                                                    # editable=True,
                                                    filter_action="native",
                                                    sort_action="native",
                                                    sort_mode="multi",
                                                    # row_selectable="multi",
                                                    # selected_rows=[],
                                                     )
                    )
                ])
            ]),


        ]),
    ])

def prepare_ml_exp_callbacks(app):
    app.ml_experiments = []
    app.aggregated_chain_results ={}

    @app.callback(
        [Output('chains-table', "selected_rows"),],
        [Input('select-all-button', 'n_clicks_timestamp'),
         Input('unselect-all-button', 'n_clicks_timestamp')],
        [State('chains-table', "derived_virtual_data"),
         ]
    )
    def unselect_all(select_timestamp,unselect_timestamp, rows):
        if select_timestamp is None and unselect_timestamp is None:
            return [[]]
        res = "SELECT"
        if unselect_timestamp is None:
            res = "SELECT"
        elif select_timestamp is None:
            res = "UNSELECT"
        elif select_timestamp > unselect_timestamp:
            res = "SELECT"
        else:
            res = "UNSELECT"

        if res == "SELECT":
            return [[i for i in range(len(rows))]]
        elif res == "UNSELECT":
            return [[]]

    @app.callback(
        [
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
            if "raw" in chain_results.columns:
                aggregated_chain_results = chain_results.drop(columns=['raw'])
            else:
                aggregated_chain_results = copy.copy(chain_results)
            #aggregate the evaluations by mean and std
            mean_chain = aggregated_chain_results.groupby('chain').mean()
            std_chain = aggregated_chain_results.groupby('chain').std()

            #rename columns
            mean_chain.columns = [column+"_mean" for column in mean_chain.columns]
            std_chain.columns = [column+"_std" for column in std_chain.columns]

            aggregated_chain_results = pd.concat([mean_chain, std_chain], axis=1).applymap(round_sig)

            aggregated_chain_results= aggregated_chain_results.reset_index()

            #remove datastream agent from the chain names
            def filter_name_agent(chain):
                if '->' in chain:
                    agents = chain.split('->')[1:]
                    chain = ""
                    for id_, agent in enumerate(agents):
                        if id_ >0:
                            chain = chain +"->"+ str(agent)
                        else:
                            chain = str(agent)
                return chain
            aggregated_chain_results['chain'] = aggregated_chain_results['chain'].apply(filter_name_agent)

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
                                                 sort_action="native",
                                                 sort_mode="multi",
                                                 id="chains-table"
                                                 )
        except:
            chains_table = " "
        return [pipeline_table, chains_table]

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

        sort_col = list(aggregated_chain_results.columns[1:3])
        aggregated_chain_results = aggregated_chain_results.sort_values(by=sort_col,ascending=False)
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

            for column in list(aggregated_chain_results.columns[1:])
        ]

        return [final_graphs]
    return app

