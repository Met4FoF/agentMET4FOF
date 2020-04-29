# Agent modules
from agentMET4FOF.agents import AgentDashboard
from agentMET4FOF import agents
from agentMET4FOF import streams

# This example shows how we can run the dashboard separately on a different IP
# The dashboard will need to connect to an existing up and running AgentNetwork,
# which is running on another process or script. Otherwise there will be error messages.

modules = [agents, streams]


def run_dashboard():
    AgentDashboard(
        dashboard_modules=modules,
        dashboard_update_interval=3,
        agent_ip_addr="127.0.0.1",
        agent_port="3333",
    )


if __name__ == "__main__":
    run_dashboard()
