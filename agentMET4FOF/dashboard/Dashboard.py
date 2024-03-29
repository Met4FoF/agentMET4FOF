# -*- coding: utf-8 -*-
import socket
from threading import Thread
from time import sleep
from wsgiref.simple_server import make_server

import dash
from dash import dcc, html
from multiprocess.context import Process

from .Dashboard_Control import _Dashboard_Control


class AgentDashboard:
    """
    Class for the web dashboard which runs with the AgentNetwork object, which by default are on the same IP.
    Optional to run the dashboard on a separate IP by providing the right parameters. See example for an implementation of a separate run of dashboard to connect to an existing agent network. If there is no existing agent network, error will show up.
    An internal _Dashboard_Control object is instantiated inside this object, which manages access to the AgentNetwork.
    """

    def __init__(
        self,
        dashboard_modules=[],
        dashboard_layouts=[],
        dashboard_update_interval=3,
        max_monitors=10,
        ip_addr="0.0.0.0",
        port=8050,
        agentNetwork="127.0.0.1",
        agent_ip_addr=3333,
        agent_port=None,
        network_stylesheet=[],
        hide_default_edge=True,
        **kwargs,
    ):
        """
        Parameters
        ----------

        dashboard_modules : modules
            Modules which are separate files, and contain classes of agents to be imported into the dashboard's interactive "Add Agent" function

        dashboard_update_interval : int
            Auto refresh rate which the dashboard queries the states of Agent Network to update the graphs and display

        max_monitors : int
            Due to complexity in managing and instantiating dynamic figures, a maximum number of monitors is specified first and only the
            each Monitor Agent will occupy one of these figures. It is not ideal, but will undergo changes for the better.

        ip_addr : str
            IP Address of the dashboard to be instantiated on.

        port : int
            Port number of the dashboard to be instantianted on.

        agentNetwork: AgentNetwork
            AgentNetwork object. If `agent_ip_addr` or `agent_port` is provided, this argument will be ignored and try to connect to a new AgentNetwork specified by those arguments.

        agent_ip_addr : str
            IP Address of the Agent Network address to connect to. By default, it will match that of the ip_addr, assuming the Dashboard and Agent Network are run on the same machine with same IP address.
            If they are meant to be separated, then this argument will point to the IP address of where the Agent Network is running.

        agent_port : int
            Port of the Agent Network port. The rationale is similar to that of the argument `agent_ip_addr`.

        """
        super(AgentDashboard, self).__init__()
        if self.is_port_at_ip_available(ip_addr, port):
            if dashboard_modules is not None and dashboard_modules is not False:

                # initialise the dashboard layout and its control here
                self.ip_addr = ip_addr
                self.port = port
                self.external_stylesheets = [
                    "https://fonts.googleapis.com/icon?family=Material+Icons",
                    "https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css",
                ]
                self.external_scripts = [
                    "https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"
                ]
                self.app = self.init_app_layout(
                    update_interval_seconds=dashboard_update_interval,
                    max_monitors=max_monitors,
                    dashboard_layouts=dashboard_layouts,
                    network_stylesheet=network_stylesheet,
                    hide_default_edge=hide_default_edge,
                    **kwargs,
                )
                self.app.dashboard_ctrl = _Dashboard_Control(
                    modules=dashboard_modules,
                    agent_ip_addr=agent_ip_addr,
                    agent_port=agent_port,
                    agentNetwork=agentNetwork,
                )
                # Spawn a very simple WSGI server.
                self._server = make_server(
                    host=self.ip_addr, port=self.port, app=self.app.server
                )

        else:
            print(
                f"Dashboard or something else is running on: {ip_addr}:{port}. If "
                f"you cannot access the dashboard in your browser, try initializing "
                f"your agent network with any other port AgentNetwork([...], "
                f"port=<OTHER_PORT_THAN_{port}>)."
            )

    def run(self):
        """This is actually executed on calling start() and brings up the server"""
        if hasattr(self, "_server"):
            self._show_startup_message()
            self._server.serve_forever()

    def _show_startup_message(self):
        """This method prints the startup message of the webserver/dashboard"""
        ip_to_print = "127.0.0.1" if self.ip_addr == "0.0.0.0" else self.ip_addr
        crucial_line = (
            f"\n| visit the agentMET4FOF dashboard on http:/"
            f"/{ip_to_print}:{self.port}/ |"
        )
        crucial_line_len = len(crucial_line)

        print(
            f"\n|-".ljust(crucial_line_len - 1, "-"),
            "|\n|".ljust(crucial_line_len, " "),
            "|\n"
            f"| Your agent network is starting up. Open your browser and".ljust(
                crucial_line_len, " "
            ),
            "|",
            crucial_line,
            "\n|".ljust(crucial_line_len - 1, " "),
            f"|" f"\n|-".ljust(crucial_line_len, "-"),
            "|\n",
            sep="",
        )

    def init_app_layout(
        self,
        update_interval_seconds=3,
        max_monitors=10,
        dashboard_layouts=[],
        network_stylesheet=[],
        hide_default_edge=True,
        **kwargs,
    ):
        """
        Initialises the overall dash app "layout" which has two sub-pages (Agent network and ML experiment)

        Parameters
        ----------
        update_interval_seconds : float or int
            Auto refresh rate which the app queries the states of Agent Network to update the graphs and display

        max_monitors : int
            Due to complexity in managing and instantiating dynamic figures, a maximum number of monitors is specified first and only the
            each Monitor Agent will occupy one of these figures. It is not ideal, but will undergo changes for the better.

        Returns
        -------
        app : Dash app object
        """
        app = dash.Dash(
            __name__,
            external_stylesheets=self.external_stylesheets,
            external_scripts=self.external_scripts,
        )
        app.network_stylesheet = network_stylesheet
        app.update_interval_seconds = update_interval_seconds
        app.num_monitors = max_monitors
        app.hide_default_edge = hide_default_edge

        for key in kwargs.keys():
            setattr(app, key, kwargs[key])

        # initialise dashboard layout objects
        self.dashboard_layouts = [
            dashboard_layout(app) for dashboard_layout in dashboard_layouts
        ]

        app.layout = html.Div(
            children=[
                # header
                html.Nav(
                    [
                        html.Div(
                            [
                                html.A(
                                    "Met4FoF Agent Testbed",
                                    className="brand-logo center",
                                ),
                                html.Ul([], className="right hide-on-med-and-down"),
                            ],
                            className="nav-wrapper container",
                        )
                    ],
                    className="light-blue lighten-1",
                ),
                dcc.Tabs(
                    id="main-tabs",
                    value="agt-net",
                    children=[
                        dashboard_layout.dcc_tab
                        for dashboard_layout in self.dashboard_layouts
                    ],
                ),
                html.Div(
                    id="page-div",
                    children=[
                        dashboard_layout.get_layout()
                        for dashboard_layout in self.dashboard_layouts
                    ],
                ),
            ]
        )

        for dashboard_layout in self.dashboard_layouts:
            dashboard_layout.prepare_callbacks(app)

        @app.callback(
            [dash.dependencies.Output("page-div", "children")],
            [dash.dependencies.Input("main-tabs", "value")],
        )
        def render_content(tab):
            for dashboard_layout in self.dashboard_layouts:
                if dashboard_layout.id == tab:
                    return [dashboard_layout.get_layout()]

        return app

    def is_port_at_ip_available(self, ip_addr: str, _port: int) -> bool:
        """Check if desired port at ip is available"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Set timeout to wait for response on ip:port.
            sock.settimeout(1)
            # Check if connection is possible and shutdown checking connection.
            # Presumably this means, that a dashboard is running.
            if sock.connect_ex((ip_addr, _port)) == 0:
                sock.shutdown(socket.SHUT_RDWR)
                return False
            # Seems as if, we can actually start our dashboard server.
            return True


class AgentDashboardProcess(AgentDashboard, Process):
    """Represents an agent dashboard for the osBrain backend"""

    def terminate(self):
        """This is shutting down the application server serving the web interface"""
        super(AgentDashboardProcess, self).terminate()
        self._server.server_close()


class AgentDashboardThread(AgentDashboard, Thread):
    """Represents an agent dashboard for the Mesa backend"""

    def __init__(
        self,
        dashboard_modules=[],
        dashboard_layouts=[],
        dashboard_update_interval=3,
        max_monitors=10,
        ip_addr="127.0.0.1",
        port=8050,
        agentNetwork="127.0.0.1",
        agent_ip_addr=3333,
        agent_port=None,
        **kwargs,
    ):
        super(AgentDashboardThread, self).__init__(
            dashboard_modules=dashboard_modules,
            dashboard_layouts=dashboard_layouts,
            dashboard_update_interval=dashboard_update_interval,
            max_monitors=max_monitors,
            ip_addr=ip_addr,
            port=port,
            agentNetwork=agentNetwork,
            agent_ip_addr=agent_ip_addr,
            agent_port=agent_port,
            **kwargs,
        )
        # Make sure, we are actually able to stop the server running from outside by
        # calling terminate() later.
        self._supposed_to_run = True

    def run(self):
        """This is actually executed on calling start() and brings up the server"""
        if hasattr(self, "_server"):
            super().run()
            # Make sure, we are actually able to stop the server running from outside.
            while not self._supposed_to_run:
                sleep(9)
            return 0
        return 1

    def terminate(self):
        """This is shutting down the application server serving the web interface"""
        try:
            self._server.shutdown()
            self._server.server_close()
        except AttributeError:
            # In this case the dashboard has in fact already been shutdown earlier.
            pass
        self._supposed_to_run = False
