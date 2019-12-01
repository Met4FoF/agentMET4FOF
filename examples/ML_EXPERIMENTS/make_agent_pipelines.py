#!/usr/bin/env python
# coding: utf-8

# ## Goal

# In[1]:


# #option 1 : group of pipelines with 3 levels
# ML_pipelines_A = make_agent_pipelines([PCA(), KNN()],
#                                 [StandardScaler(),RobustScaler()],
#                                 [LinearRegression(),ANN()], parameters)

# #option 2 : multi pipelines of single level
# ML_pipelines_B = make_agent_pipelines([CNN(),BCNN(),ANN()], parameters)

# #option 3 : single pipeline
# ML_pipeline_C = make_agent_pipeline([StandardScaler(),PCA(),ANN()], parameters)


# In[2]:


import os
path = "F:/PhD Research/Github/develop_ml_experiments_met4fof/agentMET4FOF"
os.chdir(path)

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent

from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# In[3]:


class ANN():
    def transform(self):
        return 12333


# In[4]:


method_pca = PCA()
method_ann = ANN()

print(str(type(method_pca.fit)))
print(str(type(method_ann.transform)))


# In[5]:


class TransformerAgent(AgentMET4FOF):
    def init_parameters(self,method=None):
        self.method = method
        
    def on_received_message(self,message):
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
                self.method = self.method.fit(X,Y)
                if hasattr(self.method,'predict'):
                    return 0
                
        #proceed in transforming or predicting
        if hasattr(self.method, 'transform'):
            results = self.method.transform(X)
        elif hasattr(self.method, 'predict'):
            results = self.method.predict(X)
        else: #it is a plain function
            results = self.method(X)
            
        #update chain
        chain = self.record_chain(message)
        
        #send out
        #if it is a base model, don't send out the predicted train results
        if hasattr(self.method, 'predict'):
            self.send_output({'x':X, 'y_true':Y, 'y_pred':results,'chain':chain},channel=message['channel'])
        else:
            self.send_output({'x':results, 'y':Y,'chain':chain},channel=message['channel'])
            
    def record_chain(self,message):
        chain =[]
        if type(message['data']) == dict and 'chain' in message['data'].keys():
            chain = message['data']['chain']
        else:
            chain = [message['from']]
        chain.append(self.name)
        return chain


# In[6]:


class EvaluationAgent(AgentMET4FOF):
    def init_parameters(self,method=None, **kwargs):
        self.method = method
        self.eval_params = kwargs
    def on_received_message(self,message):
        #only evaluate if it is not train channel
        if message['channel'] != 'train':
            #temporary fix
            results = self.method(message['data']['y_true'],message['data']['y_pred'], **self.eval_params)
            if type(message['data']) == dict and 'chain' in message['data'].keys():
                chain = message['data']['chain']
                agent_string = ""
                for index, agent in enumerate(chain):
                    if index !=0:
                        agent_string = agent_string +'->'+ agent
                    else:
                        agent_string = agent_string+agent
                agent_string = agent_string+'->'+self.method.__name__
                self.send_output({agent_string:results})
            else:
                self.send_output({self.method.__name__:results})


# In[7]:


class DataStreamAgent(AgentMET4FOF):
    def init_parameters(self, data=None):
        self.data = data
        self.kf = KFold(n_splits=5,shuffle=True)
        
    def agent_loop(self):
        if self.current_state == "Running":
            for train_index, test_index in self.kf.split(self.data.data):
                x_train, x_test = self.data.data[train_index], self.data.data[test_index]
                y_train, y_test = self.data.target[train_index], self.data.target[test_index]
                self.send_output({'x':x_train,'y':y_train},channel="train")
                self.send_output({'x':x_test,'y':y_test},channel="test")
            self.current_state = "Stop"
            


# In[8]:


class AgentPipeline:
    def __init__(self, agentNetwork=None,*argv):
        agentNetwork = agentNetwork
        self.pipeline = self.make_agent_pipelines(agentNetwork, argv)

    def make_transform_agent(self,agentNetwork, pipeline_component=None):
        if ("function" in type(pipeline_component).__name__) or ("method" in type(pipeline_component).__name__):
            transform_agent = agentNetwork.add_agent(pipeline_component.__name__+"_Agent",agentType=TransformerAgent)
            transform_agent.init_parameters(pipeline_component)
        elif "AgentMET4FOF" in type(pipeline_component).__name__:
            transform_agent = pipeline_component
        else: #class objects with fit and transform
            transform_agent = agentNetwork.add_agent(pipeline_component.__name__+"_Agent",agentType=TransformerAgent)
            transform_agent.init_parameters(pipeline_component())
        return transform_agent

    def make_agent_pipelines(self,agentNetwork=None, argv=[]):
        if agentNetwork is None:
            print("You need to pass an agent network as parameter to add agents")
            return -1
        agent_pipeline = []
        for pipeline_level, pipeline_component in enumerate(argv):
        #create the pipeline level, and the agents
            #handle list type
            agent_pipeline.append([])
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

        #now connect the agents on one level to the next levels, for every pipeline level
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
                
    def agents(self):
        agent_names = []
        for level in self.pipeline:
            agent_names.append([])
            for agent in level:
                agent_names[-1].append(agent.get_attr('name'))
        return agent_names


# In[9]:

if __name__ == '__main__':
    agentNetwork = AgentNetwork()


    # In[10]:


    datastream_agent = agentNetwork.add_agent(agentType=DataStreamAgent)

    evaluation_agent = agentNetwork.add_agent(agentType=EvaluationAgent)

    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    datastream_agent.init_parameters(datasets.load_iris())
    evaluation_agent.init_parameters(f1_score,average='micro')
    evaluation_agent.bind_output(monitor_agent)


    # In[11]:


    ML_Agent_pipelines_A = AgentPipeline(agentNetwork, [PCA],
                                    [StandardScaler,RobustScaler],
                                    [LogisticRegression,SVC])


    # In[12]:


    ML_Agent_pipelines_A.bind_output([evaluation_agent])


    # In[13]:


    datastream_agent.bind_output(ML_Agent_pipelines_A)
    datastream_agent.bind_output(ML_Agent_pipelines_A.pipeline[1])

    # In[14]:


    def joe(**kwargs):
        print(kwargs)

    joe(poi=123,eqwe=123)


    # In[15]:


    joe.__name__


    # In[ ]:




