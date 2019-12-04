from agentMET4FOF.agents import AgentMET4FOF
import pandas as pd

class EvaluationAgent(AgentMET4FOF):
    def init_parameters(self, methods=None, eval_params=[], ML_exp=False, **kwargs):
        if type(methods) is not list:
            methods = [methods]
        self.methods = methods
        self.eval_params = eval_params
        self.kwargs = kwargs
        self.ML_exp = ML_exp

    def on_received_message(self, message):
        #only evaluate if it is not train channel
        if message['channel'] != 'train':
            results = {}
            for method_id, method in enumerate(self.methods):
                if len(self.eval_params) !=0:
                    results.update({str(method.__name__):method(message['data']['y_true'], message['data']['y_pred'], **self.eval_params[method_id])})
                elif len(self.methods) == 1 and len(self.eval_params) == 0:
                    results.update({str(method.__name__):method(message['data']['y_true'], message['data']['y_pred'], **self.kwargs)})
                else:
                    results.update({str(method.__name__):method(message['data']['y_true'], message['data']['y_pred'])})
            if type(message['data']) == dict and 'chain' in message['data'].keys():
                agent_chain = message['data']['chain']
                for key in results.keys():
                    self.send_output({agent_chain+"-"+key: results[key]})

            else:
                self.send_output(results)

            if self.ML_exp:
                log_results = {"chain":agent_chain, "raw": pd.DataFrame.from_dict({'y_true': message['data']['y_true'],'y_pred':message['data']['y_pred']})}
                log_results.update(results)
                self.log_ML(log_results)


