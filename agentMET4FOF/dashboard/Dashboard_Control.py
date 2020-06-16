import networkx as nx

from .. import agents as agentmet4fof_module
from .. import streams as datastreammet4fof_module


#global variables access via 'dashboard_ctrl'
class _Dashboard_Control():
    def __init__(self, agent_ip_addr="127.0.0.1", agent_port=3333, modules= [], agentNetwork=None):
        super(_Dashboard_Control, self).__init__()
        self.network_layout = {'name': 'grid'}
        self.current_selected_agent = " "
        self.current_nodes = []
        self.current_edges = []
        # get nameserver
        self.agent_graph = nx.DiGraph()
        if agentNetwork:
            self.agentNetwork = agentNetwork
        else:
            self.agentNetwork = agentmet4fof_module.AgentNetwork(ip_addr=agent_ip_addr,port=agent_port, connect=True, dashboard_modules=False) #dashboard_modules has to be false, to prevent infinite loop
        if isinstance(modules, bool) and modules:
            modules = []
        elif isinstance(modules, list):
            modules = modules
        elif type(modules).__name__ == "module":
            modules = [modules]
        self.modules = [agentmet4fof_module, datastreammet4fof_module] + modules

    def get_agentTypes(self):
        agentTypes ={}
        for module_ in self.modules:
            agentTypes.update(dict([(name, cls) for name, cls in module_.__dict__.items() if
                               isinstance(cls, type) and cls.__bases__[-1] == agentmet4fof_module.AgentMET4FOF]))
        agentTypes.pop("_AgentController",None)
        agentTypes.pop("_Logger",None)
        agentTypes.pop("DataStreamAgent",None)
        return agentTypes

    def get_datasets(self):
        datasets ={}
        for module_ in self.modules:
            datasets.update(dict([(name, cls) for name, cls in module_.__dict__.items() if
                               isinstance(cls, type) and cls.__bases__[-1] == datastreammet4fof_module.DataStreamMET4FOF]))
        return datasets
