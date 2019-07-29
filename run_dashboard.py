from AgentMET4FOF import run_dashboard

#Agent modules
import develop.develop_zema_agents as zema_agents
import develop.develop_zema_datastream as zema_datastream

modules = [zema_agents, zema_datastream]

if __name__ == "__main__":
    run_dashboard(dashboard_modules=modules,dashboard_update_interval=3)
