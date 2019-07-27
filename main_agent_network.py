from AgentMET4FOF import AgentNetwork
from dashboard.Dashboard import Dashboard_Control, app

#Agent modules
import develop.develop_zema_agents as zema_agents

modules = [zema_agents]

if __name__ == '__main__':

    #start agent network server
    #agentNetwork = AgentNetwork(visualize = [modules])
    agentNetwork = AgentNetwork()
    # dashboard_ctrl = Dashboard_Control(modules)
    # app.dashboard_ctrl = dashboard_ctrl
    # app.run_server(debug=False)
