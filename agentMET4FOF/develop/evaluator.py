from agentMET4FOF.agents import AgentMET4FOF
import pandas as pd

class EvaluationAgent(AgentMET4FOF):
    def init_parameters(self,method=None, ML_exp=False, **kwargs):
        self.method = method
        self.eval_params = kwargs
        self.ML_exp = ML_exp

    def on_received_message(self,message):
        #only evaluate if it is not train channel
        if message['channel'] != 'train':
            results = self.method(message['data']['y_true'],message['data']['y_pred'], **self.eval_params)
            if type(message['data']) == dict and 'chain' in message['data'].keys():
                agent_chain = message['data']['chain']
                self.send_output({agent_chain:results})


            else:
                agent_chain = self.method.__name__
                self.send_output({self.method.__name__:results})


            if self.ML_exp:
                self.log_ML({agent_chain: {"evaluation":results,
                                           "raw": pd.DataFrame.from_dict({'y_true': message['data']['y_true'],'y_pred':message['data']['y_pred']})
                                           }
                             })

