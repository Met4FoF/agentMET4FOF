from osbrain.logging import Logger
from agentMet4FoF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMet4FoF.streams import SineGenerator
import re
import pandas as pd
import sys

class _Logger(AgentMET4FOF):

    def init_parameters(self,log_filename= "log_file.xlsx", save_logfile=True):
        self.bind('SUB', 'sub', self.log_handler)
        self.logs = pd.DataFrame.from_dict({'Time':[],'Name':[],'Topic':[],'Data':[]})
        self.log_filename = log_filename

        try:
            self.logs.to_excel(self.log_filename)
        except:
            raise Exception

    def log_handler(self, message, topic):
        #print message to console
        sys.stdout.write(message)
        sys.stdout.flush()

        self.transform_log_info(str(message))

    def transform_log_info(self, log_msg):
        re_sq = '\[(.*?)\]'
        re_rd = '\((.*?)\)'

        date = re.findall(re_sq,log_msg)[0]
        agent_name = re.findall(re_rd,log_msg)[0]

        contents = log_msg.split(':')
        if len(contents) > 4:
            topic = contents[3]
            data = contents[4:]
        else:
            topic = contents[3]
            data = " "

        new_log_df = pd.DataFrame({'Time':date,'Name':agent_name,'Topic':topic,'Data':data},index=[0])
        self.logs = self.logs.append(new_log_df).reset_index(drop=True)
        try:
            self.logs.to_excel(self.log_filename)
        except:
            raise Exception

class SineGeneratorAgent(AgentMET4FOF):
    def init_parameters(self):
        self.stream = SineGenerator()

    def agent_loop(self):
        if self.current_state == "Running":
            sine_data = self.stream.next_sample() #dictionary
            self.send_output(sine_data['x'])

if __name__ == '__main__':
    #start agent network server
    agentNetwork = AgentNetwork(dashboard_modules=False)

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType= SineGeneratorAgent)
    _logger_agent = agentNetwork.add_agent(agentType= _Logger)
    monitor_agent = agentNetwork.add_agent(agentType= MonitorAgent)

    gen_agent.set_logger(_logger_agent)

    #connect agents by either way:
    # 1) by agent network.bind_agents(source,target)
    agentNetwork.bind_agents(gen_agent, monitor_agent)

    # 2) by the agent.bind_output()
    gen_agent.bind_output(monitor_agent)

    # set all agents states to "Running"
    agentNetwork.set_running_state()



