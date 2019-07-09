#Agent dependencies
from osbrain import run_nameserver
from osbrain import run_agent
from osbrain import Agent
from osbrain import NSProxy

#ML dependencies
import numpy as np
from skmultiflow.data import WaveformGenerator
from skmultiflow.trees import HoeffdingTree
from sklearn.model_selection import StratifiedKFold

class AgentMET4FOF(Agent):
    def on_init(self):
        self.Inputs = {}
        self.Outputs = {}
        self.PubAddr_alias = self.name + "_PUB"
        self.PubAddr = self.bind('PUB', alias=self.PubAddr_alias)
        self.AgentType = type(self).__name__
        self.log_info("INITIALIZED")
        self.states = {0: "Idle", 1: "Running", 2: "Pause", 3: "Stop"}
        self.current_state = self.states[0]
        self.init_parameters()

    def init_parameters(self):
        return 0

    def before_loop(self):
        self.init_agent_loop(1.0)

    def init_agent_loop(self, loop_wait=1.0):
        self.stop_all_timers()
        #check if agent_loop is overriden by user
        if self.__class__.agent_loop == AgentMET4FOF.agent_loop:
            return 0
        else:
            self.each(loop_wait, self.__class__.agent_loop)
        return 0

    def agent_loop(self):
        return 0

    def on_received_message(self, message):
        return message

    def pack_data(self,data):
        return {"from":self.name,"data":data}

    def send_output(self, data):
        self.send(self.PubAddr, self.pack_data(data), topic='data')

        # LOGGING
        if self.log_mode:
            self.log_info("Sending: "+str(data))

        return data

    def handle_process_data(self, message):
        # LOGGING
        if self.log_mode:
            self.log_info("Received: "+str(message))
        # process the received data here
        proc_msg = self.on_received_message(message)

        return proc_msg

    def bind_output(self, output_module):
        output_module_id = output_module.get_attr('name')
        if output_module_id not in self.Outputs:
            # update self.Outputs list and Inputs list of output_module
            self.Outputs.update({output_module.get_attr('name'): output_module})
            temp_updated_inputs= output_module.get_attr('Inputs')
            temp_updated_inputs.update({self.name: self})
            output_module.set_attr(Inputs=temp_updated_inputs)
            # bind to the address
            if output_module.has_socket(self.PubAddr_alias):
                output_module.subscribe(self.PubAddr_alias, handler={'data': AgentMET4FOF.handle_process_data})
            else:
                output_module.connect(self.PubAddr, alias=self.PubAddr_alias, handler={'data':AgentMET4FOF.handle_process_data})

            # LOGGING
            if self.log_mode:
                self.log_info("Connected output module"+ output_module_id)

    def unbind_output(self, target_module):
        module_id = target_module.get_attr('name')
        if module_id in self.Outputs:
            self.Outputs.pop(module_id, None)
            target_module.get_attr('Inputs').pop(self.name, None)
            target_module.unsubscribe(self.PubAddr_alias,'data')

            # LOGGING
            if self.log_mode:
                self.log_info("Disconnected output module: "+ module_id)
    def print_sockets(self):
        self.log_info(self.socket)

class AgentController(AgentMET4FOF):
    def init_parameters(self, ns=None):
        self.ns = ns

    def get_agentType_count(self, agentType):
        num_count = 1
        agentType_name = str(agentType.__name__)
        if len(self.ns.agents()) != 0 :
            for agentName in self.ns.agents():
                current_agent_type = self.ns.proxy(agentName).get_attr('AgentType')
                if current_agent_type == agentType_name:
                    num_count+=1
        return num_count

    def generate_module_name(self, agentType):
        name = agentType.__name__
        name += "_"+str(self.get_agentType_count(agentType))
        return name

    def add_module(self, name=" ", agentType= AgentMET4FOF, log_mode=True):
        if name == " ":
            name= self.generate_module_name(agentType)
        return run_agent(name, base=agentType, attributes=dict(log_mode=True), nsaddr=self.ns.addr())

    def agents(self):
        exclude_names = ["AgentController"]
        agent_names = [name for name in self.ns.agents() if name not in exclude_names]
        return agent_names

#global control
class AgentNetwork():
    def __init__(self):
        self.states = {0: "Idle", 1: "Running", 2: "Pause", 3: "Stop"}
        self.current_state = "Idle"
        self.controller = None
    def connect(self, port = 3333):
        try:
            self.ns = NSProxy(nsaddr='127.0.0.1:' + str(port))
        except:
            print("Unable to connect to existing NameServer...")
            self.ns = 0

    def start_server(self, port = 3333):
        print("Starting NameServer...")
        self.ns = run_nameserver(addr='127.0.0.1:' + str(port))
        if len(self.ns.agents()) != 0:
            self.ns.shutdown()
            self.ns = run_nameserver(addr='127.0.0.1:' + str(port))
        controller= run_agent("AgentController", base=AgentController, attributes=dict(log_mode=True), nsaddr=self.ns.addr())
        controller.init_parameters(self.ns)

    def set_mode(self, state):
        self.current_state = state
    def get_mode(self):
        return self.current_state

    def set_running_state(self, filter_agent=None):
        self.set_agents_state(filter_agent=filter_agent,state="Running")

    def set_stop_state(self, filter_agent=None):
        self.set_agents_state(filter_agent=filter_agent, state="Stop")

    def set_agents_state(self, filter_agent=None, state="Idle"):
        self.set_mode(state)
        for agent_name in self.agents():
            if (filter_agent is not None and filter_agent in agent_name) or (filter_agent is None):
                agent = self.get_agent(agent_name)
                agent.set_attr(current_state = state)
        print("SET STATE:  ", state)
        return 0

    def bind_agents(self, source, target):
        source.bind_output(target)
        return 0

    def unbind_agents(self, source, target):
        source.unbind_output(target)
        return 0

    def get_controller(self):
        if self.controller is None:
            self.controller = self.ns.proxy('AgentController')
        return self.controller

    def get_agent(self,agent_name):
        return self.get_controller().get_attr('ns').proxy(agent_name)

    def agents(self):
        #exclude_names = ["AgentController"]
        #agent_names = [name for name in self.ns.agents() if name not in exclude_names]
        agent_names = self.get_controller().agents()
        return agent_names

    def add_agent(self, name=" ", agentType= AgentMET4FOF, log_mode=True):
        agent = self.get_controller().add_module(name=name, agentType= agentType, log_mode=log_mode)
        #agent = run_agent(base=agentType, attributes=dict(log_mode=True), nsaddr=self.ns.addr())
        return agent

    def shutdown(self):
        self.get_controller().get_attr('ns').shutdown()
        return 0

class DataStream(AgentMET4FOF):

    def init_parameters(self, n_wait=1.0, stream = WaveformGenerator(),pretrain_size = 100, max_samples = 100000, batch_size=100 ):

        # parameters
        # setup data stream
        self.stream = stream
        self.stream.prepare_for_use()
        self.pretrain_size = pretrain_size
        self.max_samples = max_samples
        self.batch_size = batch_size

        self.current_sample = 0
        self.first_time = True


    def agent_loop(self):
        #if is running
        if self.current_state == self.states[1]:
            data = self.read_data()
            if data is not None:
                self.send_output(data)
        else:
            return 0

    def read_data(self):
        if self.current_sample < self.max_samples:
            # get sample
            if (self.first_time):
                data = self.stream.next_sample(self.pretrain_size)
                self.current_sample += self.pretrain_size
                self.first_time = False
            else:
                data = self.stream.next_sample(self.batch_size)
                self.current_sample += self.batch_size
        else:
            data = None

        #log
        self.log_info(data)
        return data

class ML_Model(AgentMET4FOF):
    def init_parameters(self, mode="prequential", ml_model= HoeffdingTree(), split_type=None):
        self.mode = mode
        self.ml_model = ml_model
        self.results = []
        if split_type is not None:
            self.split_type = split_type
        else:
            self.split_type = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    def on_received_message(self, message):
        if len(message) >1:
            x = message['data'][0]
            y = message['data'][1]
        else:
            return -1

        # prequential: test & train
        if self.mode == "prequential":
            y_pred = self.ml_model.predict(x)
            self.ml_model.partial_fit(x, y)
            res = self.compute_accuracy(y_pred=y_pred, y_true=y)
            self.results.append(res)

        # prequential: test & train
        elif (self.mode == "holdout"):
            res_temp = []
            # begin kfold
            for train_index, test_index in self.split_type.split(x, y):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self.ml_model.partial_fit(x_train, y_train)
                y_pred = self.ml_model.predict(x_test)
                res = self.compute_accuracy(y_pred=y_pred, y_true=y_test)
                res_temp.append(res)
            self.results.append(np.mean(res_temp))

        self.send_output(self.results[-1])

    # classifier accuracy - user defined
    def compute_accuracy(self, y_pred, y_true):
        res = y_pred == y_true
        num_accurate = [1 if y == True else 0 for y in res]
        accuracy = np.sum(num_accurate) / len(num_accurate) * 100
        return accuracy

class MonitorAgent(AgentMET4FOF):
    def init_parameters(self):
        #dictionary of Inputs
        self.memory = {}

    def on_received_message(self, message):
        self.log_info(message)
        self.log_info(self.memory)
        if message['from'] in self.memory:
            self.memory[message['from']].append(message['data'])
        else:
            self.memory.update({message['from']:[message['data']]})
        return message