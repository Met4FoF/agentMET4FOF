from ..agents import AgentMET4FOF
from sklearn.model_selection import KFold


class DataStreamAgent(AgentMET4FOF):
    def init_parameters(self, data_name="unnamed_data", train_mode={"Prequential","Kfold5","Kfold10"}, x=None, y=None):
        #if data_name is provided, will assign to pre-made datasets,
        #otherwise for custom dataset, three parameters : data_name, x and y will need to be assigned
        self.data_name = data_name
        self.x = x
        self.y = y
        if type(train_mode) == set:
            self.train_mode = "Kfold5"

    def agent_loop(self):
        if self.current_state == "Running":
            if self.train_mode == "Kfold5":
                self.kf = KFold(n_splits=5,shuffle=True)
                for train_index, test_index in self.kf.split(self.x):
                    x_train, x_test = self.x[train_index], self.x[test_index]
                    y_train, y_test = self.y[train_index], self.y[test_index]
                    self.send_output({'x':x_train,'y':y_train},channel="train")
                    self.send_output({'x':x_test,'y':y_test},channel="test")
            self.current_state = "Stop"
