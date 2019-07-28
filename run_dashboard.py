import dashboard.Dashboard
import dashboard.Dashboard_Control as Dashboard_Control

#Agent modules
import develop.develop_zema_agents as zema_agents
import develop.develop_zema_datastream as zema_datastream

modules = [zema_agents, zema_datastream]

if __name__ == "__main__":
    # get nameserver
    dashboard_ctrl = Dashboard_Control(modules=modules)
    dashboard.Dashboard.app.dashboard_ctrl = dashboard_ctrl
    dashboard.Dashboard.app.run_server(debug=False)
