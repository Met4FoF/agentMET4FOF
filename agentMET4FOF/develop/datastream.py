from agentMET4FOF.agents import AgentMET4FOF
from sklearn.model_selection import KFold

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
