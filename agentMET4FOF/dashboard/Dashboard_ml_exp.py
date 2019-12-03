import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash
import os
import time
import dash_table
from dash.exceptions import PreventUpdate
import agentMET4FOF.dashboard.LayoutHelper as LayoutHelper
from agentMET4FOF.dashboard.LayoutHelper import create_nodes_cytoscape, create_edges_cytoscape, create_monitor_graph
from datetime import datetime
import pandas as pd

experiment_list = []
experiment_base_path ="F:/PhD Research/Github/develop_ml_experiments_met4fof/agentMET4FOF/examples/ML_EXPERIMENTS/"
experiment_folder = "ML_EXP"
experiment_path = experiment_base_path+experiment_folder
print(os.listdir(experiment_path))
os.chdir(experiment_path)
for dir_ in os.listdir(experiment_base_path+experiment_folder):
    if "." not in dir_:
        date_mod = datetime.strptime(time.ctime(os.path.getmtime(dir_)), "%a %b %d %H:%M:%S %Y")
        date_mod_string = date_mod.strftime("%d-%m-%Y, %H:%M")
        experiment_list.append({"Name":dir_,"__date_time":date_mod, "Date": date_mod_string})
        # print(dir_+":  last modified: %s" % time.ctime(os.path.getmtime(dir_)))

df = pd.DataFrame(experiment_list).sort_values(by=["__date_time","Name"], ascending=False)

df = df[["Name","Date"]]
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

def get_ml_exp_layout():
    return html.Div(className="row",children=[
        #main panel
                html.Div(className="col s8", children=[
                        html.Div(className="card", children=[
                           html.Div(className="card-content", children=[
                                   # html.Span(className="card-title", children=["Agent Network"]),
                                   # html.P(children=["Active agents running in agent network"]),
                                   html.Div(className="row", children = [


                            ]),
                           html.Div(className="card-action", children=[


                            ])
                        ]),
                    ]),
                ]),
                #side panel
                html.Div(className="col s4", children=[
                    html.Div(className="card blue lighten-4", children= [
                        html.Div(className="card-content", children=[

                            html.Div(style={'margin-top': '20px'}, children=[
                                html.H5(className="black-text", children="Experiments"),
                                    # LayoutHelper.create_params_table(table_name="pipeline-table",data=df,
                                    #                                     editable=True,
                                    #                                     filter_action="native",
                                    #                                     sort_action="native",
                                    #                                     sort_mode="multi",
                                    #                                     row_selectable="multi",
                                    #
                                    #
                                    #                                     selected_rows=[],
                                    #
                                    #
                                    #
                                    #                                  )
                                dash_table.DataTable(
                                    id='datatable-interactivity',
                                    columns=[
                                        {"name": i, "id": i, "selectable": True} for i in df.columns
                                    ],
                                    data=df.to_dict('records'),
                                    editable=True,
                                    filter_action="native",
                                    sort_action="native",
                                    sort_mode="multi",
                                    row_selectable="multi",
                                    selected_columns=[],
                                    selected_rows=[],
                                    page_action="native",
                                    page_current= 0,
                                    page_size= 10,
                                ),
                            ]),

                            html.Div(style={'margin-top': '20px'}, children=[

                            ])

                        ])

                    ]),

                    html.Div(className="card green lighten-4", children=[
                        html.Div(className="card-content", children=[
                            html.Div(style={'margin-top': '20px'}, children=[
                                html.H5(className="black-text", children="Pipelines"),
                            ])
                        ])
                    ])
                ]),
        ])



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css',
'https://fonts.googleapis.com/icon?family=Material+Icons']
# external_stylesheets= ['https://fonts.googleapis.com/icon?family=Material+Icons']
# external_stylesheets = ['https://github.com/plotly/dash-app-stylesheets/blob/master/dash-analytics-report.css']
external_scripts = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js']
# external_scripts=[]
assets_url_path = os.path.join(os.path.dirname(__file__), 'assets')

app = dash.Dash(__name__,
                assets_url_path=assets_url_path,
                external_stylesheets=external_stylesheets,
                external_scripts=external_scripts
                )

# app = dash.Dash(__name__)

app.layout = get_ml_exp_layout()
# external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
# for css in external_css:
#     app.css.append_css({"external_url": css})
#
# external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
# for js in external_js:
#     app.scripts.append_script({'external_url': js})

if __name__ == '__main__':
    app.run_server(debug=True)
