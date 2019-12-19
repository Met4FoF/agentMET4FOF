#Agent modules
import examples.ZEMA_EMC.zema_agents as zema_agents
import examples.ZEMA_EMC.zema_datastream as zema_datastream
from agentMET4FOF.agents import AgentDashboard

#This example shows how we can run the dashboard separately on a different IP
#The dashbaord will need to connect to an existing up and running AgentNetwork, which is running on another process or script
#Otherwise there will be error messages

modules = [zema_agents, zema_datastream]

if __name__ == "__main__":
    AgentDashboard(dashboard_modules=modules,dashboard_update_interval=3, agent_ip_addr="127.0.0.1", agent_port="3333")
