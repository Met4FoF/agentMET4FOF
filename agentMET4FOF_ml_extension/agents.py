from agentMET4FOF.agents import AgentMET4FOF
from sklearn.model_selection import ParameterGrid
import copy

class TransformerAgent(AgentMET4FOF):
    def init_parameters(self,method=None, **kwargs):
        if ("function" in type(method).__name__) or ("method" in type(method).__name__):
            self.method = method
        else:
            self.method = method(**kwargs)
        self.models = {}
        self.hyperparams = kwargs
        #for single functions passed to the method
        for key in kwargs.keys():
            self.set_attr(key=kwargs[key])

    def on_received_message(self,message):
        #update chain
        chain = self.get_chain(message)

        #if data is dict, with 'x' in keys
        if type(message['data']) == dict and 'x' in message['data'].keys():
            X= message['data']['x']
            Y= message['data']['y']
        else:
            X=message['data']
            Y=None

        #if it is train, and has fit function, then fit it first.
        if message['channel'] == 'train':
            if hasattr(self.method, 'fit'):
                self.models.update({chain:copy.deepcopy(self.method).fit(X,Y)})
                if hasattr(self.method,'predict'):
                    return 0

        #proceed in transforming or predicting
        if hasattr(self.method, 'transform'):
            if chain in self.models.keys():
                results = self.models[chain].transform(X)
            else:
                results = self.method.transform(X)
        elif hasattr(self.method, 'predict'):
            if chain in self.models.keys():
                results = self.models[chain].predict(X)
            else:
                results = self.method.predict(X)
        else: #it is a plain function
            results = self.method(X, **self.hyperparams)

        #send out
        #if it is a base model, don't send out the predicted train results
        chain= chain+"->"+self.name
        if hasattr(self.method, 'predict'):
            #check if uncertainty is included
            if type(results) == tuple and len(results) == 2:
                y_pred = results[0]
                y_unc = results[1]
                self.send_output({'x':X, 'y_true':Y, 'y_pred':y_pred,'y_unc':y_unc,'chain':chain},channel=message['channel'])
            else:
                self.send_output({'x':X, 'y_true':Y, 'y_pred':results,'chain':chain},channel=message['channel'])
        else:
            self.send_output({'x':results, 'y':Y,'chain':chain},channel=message['channel'])

    def get_chain(self,message):
        chain =""
        if type(message['data']) == dict and 'chain' in message['data'].keys():
            chain = message['data']['chain']
        else:
            chain = message['from']
        return chain

class AgentPipeline:
    def __init__(self, agentNetwork=None,*argv, hyperparameters=None):

        # list of dicts where each dict is of (key:list of hyperparams)
        # each hyperparam in the dict will be spawned as an agent
        agentNetwork = agentNetwork
        self.hyperparameters = hyperparameters
        self.pipeline = self.make_agent_pipelines(agentNetwork, argv, hyperparameters)

    def make_transform_agent(self,agentNetwork, pipeline_component=None, hyperparameters={}):
        if ("function" in type(pipeline_component).__name__) or ("method" in type(pipeline_component).__name__):
            transform_agent = agentNetwork.add_agent(pipeline_component.__name__+"_Agent",agentType=TransformerAgent)
            transform_agent.init_parameters(pipeline_component,**hyperparameters)
        elif issubclass(type(pipeline_component), AgentMET4FOF):
            transform_agent = pipeline_component
            transform_agent.init_parameters(**hyperparameters)
        else: #class objects with fit and transform
            transform_agent = agentNetwork.add_agent(pipeline_component.__name__+"_Agent",agentType=TransformerAgent)
            transform_agent.init_parameters(pipeline_component,**hyperparameters)
        return transform_agent

    def make_agent_pipelines(self,agentNetwork=None, argv=[], hyperparameters=None):
        if agentNetwork is None:
            print("You need to pass an agent network as parameter to add agents")
            return -1
        agent_pipeline = []

        if hyperparameters is not None and len(hyperparameters) == 1:
            if type(hyperparameters[0]) != list:
                hyperparameters = [hyperparameters]

        for pipeline_level, pipeline_component in enumerate(argv):
            agent_pipeline.append([])
            #create an agent for every unique hyperparameter combination
            if hyperparameters is not None:
                try:
                    if pipeline_level < len(hyperparameters):
                        hyper_param_level = hyperparameters[pipeline_level]
                    else:
                        hyper_param_level = {}
                except:
                    print("Error getting hyperparameters mapping")
                    return -1

                #now, hyper_param_level is a list of dictionaries of hyperparams for agents at pipeline_level
                if type(pipeline_component) == list:
                    for function_id ,pipeline_function in enumerate(pipeline_component):
                        print(hyper_param_level)
                        if (type(hyper_param_level) == dict) or ((len(hyper_param_level)>0) and (len(hyper_param_level[function_id]) > 0)):
                            if type(hyper_param_level) == dict:
                                param_grid = list(ParameterGrid(hyper_param_level))
                            else:
                                param_grid = list(ParameterGrid(hyper_param_level[function_id]))
                            print("MAKING {} AGENTS WITH HYPERPARAMS: {}".format(pipeline_function.__name__, param_grid))
                            for param in param_grid:
                                transform_agent = self.make_transform_agent(agentNetwork,pipeline_function,param)
                                agent_pipeline[-1].append(transform_agent)

                        else:
                            print("MAKING {} AGENT WITH DEFAULT HYPERPARAMS".format(pipeline_function.__name__))
                            #fill up the new empty list with a new agent for every pipeline function
                            transform_agent = self.make_transform_agent(agentNetwork,pipeline_function)
                            agent_pipeline[-1].append(transform_agent)

                #non list, single function, class, or agent
                else:
                    #fill up the new empty list with a new agent for every pipeline function
                    transform_agent = self.make_transform_agent(agentNetwork,pipeline_component)
                    agent_pipeline[-1].append(transform_agent)


            #otherwise there's no hyperparameters usage, proceed with defaults for all agents
            #the logic similar to before but without hyperparams loop
            else:
                if type(pipeline_component) == list:
                    for pipeline_function in pipeline_component:
                        #fill up the new empty list with a new agent for every pipeline function
                        transform_agent = self.make_transform_agent(agentNetwork,pipeline_function)
                        agent_pipeline[-1].append(transform_agent)
                #non list, single function, class, or agent
                else:
                    #fill up the new empty list with a new agent for every pipeline function
                    transform_agent = self.make_transform_agent(agentNetwork,pipeline_component)
                    agent_pipeline[-1].append(transform_agent)

        #now bind the agents on one level to the next levels, for every pipeline level
        for pipeline_level, _ in enumerate(agent_pipeline):
            if pipeline_level != (len(agent_pipeline)-1):
                for agent in agent_pipeline[pipeline_level]:
                    for agent_next in agent_pipeline[pipeline_level+1]:
                        agent.bind_output(agent_next)
        return agent_pipeline

    def bind_output(self, output_agent):
        pipeline_last_level = self.pipeline[-1]
        if "AgentPipeline" in str(type(output_agent).__name__):
            for agent in pipeline_last_level:
                for next_agent in output_agent.pipeline[0]:
                    agent.bind_output(next_agent)
        elif type(output_agent) == list:
            for agent in pipeline_last_level:
                for next_agent in output_agent:
                    agent.bind_output(next_agent)
        else:
            for agent in pipeline_last_level:
                agent.bind_output(output_agent)

    def unbind_output(self, output_agent):
        pipeline_last_level = self.pipeline[-1]
        if "AgentPipeline" in str(type(output_agent).__name__):
            for agent in pipeline_last_level:
                for next_agent in output_agent.pipeline[0]:
                    agent.unbind_output(next_agent)
        elif type(output_agent) == list:
            for agent in pipeline_last_level:
                for next_agent in output_agent:
                    agent.unbind_output(next_agent)
        else:
            for agent in pipeline_last_level:
                agent.unbind_output(output_agent)

    def agents(self,ret_hyperparams=False):
        agent_names = []
        hyperparams = []
        for level in self.pipeline:
            agent_names.append([])
            hyperparams.append([])
            for agent in level:
                agent_names[-1].append(agent.get_attr('name'))
                if ret_hyperparams:
                    hyperparams[-1].append(agent.get_attr('hyperparams'))

        if ret_hyperparams:
            return {"agents":agent_names,"hyperparams":hyperparams}
        else:
            return agent_names
