from ..agents import AgentMET4FOF
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
            check_y_unc = False
            if "y_unc" in message['data'].keys():
                check_y_unc = True

            for method_id, method in enumerate(self.methods):
                if len(self.eval_params) !=0:
                    if "y_unc" in method.__code__.co_varnames and check_y_unc:
                        new_res={str(method.__name__):method(message['data']['y_true'], message['data']['y_pred'], message['data']['y_unc'], **self.eval_params[method_id])}
                    else:
                        new_res={str(method.__name__):method(message['data']['y_true'], message['data']['y_pred'], **self.eval_params[method_id])}
                elif len(self.methods) == 1 and len(self.eval_params) == 0:
                    if "y_unc" in method.__code__.co_varnames and check_y_unc:
                        new_res={str(method.__name__):method(message['data']['y_true'], message['data']['y_pred'], message['data']['y_unc'], **self.kwargs)}
                    else:
                        new_res={str(method.__name__):method(message['data']['y_true'], message['data']['y_pred'], **self.kwargs)}
                else:
                    if "y_unc" in method.__code__.co_varnames and check_y_unc:
                        new_res={str(method.__name__):method(message['data']['y_true'], message['data']['y_pred'],message['data']['y_unc'])}
                    else:
                        new_res={str(method.__name__):method(message['data']['y_true'], message['data']['y_pred'])}
                results.update(new_res)
            if type(message['data']) == dict and 'chain' in message['data'].keys():
                agent_chain = message['data']['chain']
                for key in results.keys():
                    self.send_output({agent_chain+"-"+key: results[key]})

            else:
                self.send_output(results)

            if self.ML_exp:
                if check_y_unc:
                    log_results = {"chain":agent_chain, "raw": pd.DataFrame.from_dict({'y_true': message['data']['y_true'],'y_pred':message['data']['y_pred'],'y_unc':message['data']['y_unc']})}
                else:
                    log_results = {"chain":agent_chain, "raw": pd.DataFrame.from_dict({'y_true': message['data']['y_true'],'y_pred':message['data']['y_pred']})}
                log_results.update(results)
                self.log_ML(log_results)


