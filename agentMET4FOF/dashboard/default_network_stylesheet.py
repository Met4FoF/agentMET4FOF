# For more information on all the available styles,
# Please visit: https://dash.plotly.com/cytoscape/styling

default_agent_network_stylesheet = [
    {
        "selector": "node",
        "style": {
            "label": "data(id)",
            "shape": "rectangle",
            "text-valign": "center",
            "text-halign": "center",
            "color": "#FFF",
            "text-outline-width": 1.5,
            "text-outline-color": "#000232",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style": "unbundled-bezier",
            "mid-target-arrow-shape": "triangle",
            "arrow-scale": 2,
            "line-color": "#4287f5",
            "mid-target-arrow-color": "#4287f5",
            "label": "data(channel)",
            "text-outline-width": 1.5,
            "text-outline-color": "#000232",
            "color": "#FFF",
            "text-margin-x": "10px",
            "text-margin-y": "20px",
        },
    },
    {"selector": ".rectangle", "style": {"shape": "rectangle"}},
    {"selector": ".triangle", "style": {"shape": "triangle"}},
    {"selector": ".octagon", "style": {"shape": "octagon"}},
    {"selector": ".ellipse", "style": {"shape": "ellipse"}},
    {"selector": ".bluebackground", "style": {"background-color": "#c4fdff"}},
    {"selector": ".blue", "style": {"background-color": "#006db5"}},
    {
        "selector": ".coalition",
        "style": {
            "background-color": "#c4fdff",
            "text-valign": "top",
            "text-halign": "center",
        },
    },
    {"selector": ".coalition-edge", "style": {"line-style": "dashed"}},
    {
        "selector": ".outline",
        "style": {
            "color": "#fff",
            "text-outline-color": "#888",
            "text-outline-width": 2,
        },
    },
]
