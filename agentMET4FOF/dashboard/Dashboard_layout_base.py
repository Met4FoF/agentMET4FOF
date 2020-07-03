#Abstract Base class for every `page` in the dashboard
#Note the Agent network page, and the ML_Experiment extension page are derived from this base class

import dash_html_components as html
import dash_core_components as dcc

class Dashboard_Layout_Base():
    def __init__(self, app):
        self.app = app
        self.set_layout_name()

    def set_layout_name(self,id="agt-net", title="Agent Network"):

        """
        Unique name for each layout. Should set this to be hard-coded and not intended for users to modify it.
        See `__init__` on when it is called.

        Parameters
        ----------
        id : str
            id of this layout. (e.g. agt-net for agent network page. ml-exp for ml-experiment page)
        title : str
            Display title on the top header tab of dashboard
        """
        self.id = id
        self.title=title

    @property
    def dcc_tab(self):
        return dcc.Tab(id=self.id+"-tab", value=self.id,label=self.title, children=[])

    def get_layout(self) -> html.Div:
        """
        Renders the layout page using dash plotly html.Div object

        Returns
        -------
        dash_html_components.Div to be rendered

        """
        return 0

    def prepare_callbacks(self,app):
        """
        Prepares the dash plotly callbacks to the plotly app

        Parameters
        ----------
        app : Dash app object

        """
        return 0

