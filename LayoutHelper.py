import dash
import dash_cytoscape as cyto
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go

#return icon and text for button
def html_icon(icon="play_circle_filled" ,text="Start", alignment="left"):
    return [html.I(icon, className="material-icons "+alignment),text]

def html_button(icon="play_circle_filled", text="Start",id=" "):
    return html.Button(children=html_icon(icon,text), id=id, className="btn waves-light", style={'margin-top': '10px'})