import numpy as np
import torch
from scipy import stats
from torch.autograd import Variable

from agentMET4FOF_ml_extension.ml_uncertainty.bnn_utils import parse_architecture_string


class BNN_Dropout(torch.nn.Module):
    def __init__(self, input_size, output_size, architecture=["d1","d1"], dropout_p=0.2,use_cuda=False):
        super(BNN_Dropout, self).__init__()
        self.dropout_p = dropout_p
        self.architecture = architecture
        self.use_cuda = use_cuda

        layers = parse_architecture_string(input_size,output_size, architecture, layer_type=torch.nn.Linear)
        if self.use_cuda:
            for layer_index,layer in enumerate(layers):
                layers[layer_index] = layers[layer_index]
            self.layers = torch.nn.ModuleList(layers).cuda()
            self.dropout_layer = torch.nn.Dropout(p=self.dropout_p).cuda()
        else:
            self.layers = torch.nn.ModuleList(layers)
            self.dropout_layer = torch.nn.Dropout(p=self.dropout_p)

    def forward(self,x):
        if "Tensor" not in type(x).__name__:
            x = Variable(torch.from_numpy(x).float())
        x_temp = x

        for layer_index,layer in enumerate(self.layers):
            if layer_index != (len(self.layers)-1) or layer_index ==0:
                x_temp = layer(self.dropout_layer(x_temp))
            else:
                x_temp = layer(x_temp)
        return x_temp

class BNN_Model():
    def __init__(self, method="dropout", task={"regression","classification"}, num_samples=50,num_epochs=100,
                 learning_rate=0.001, return_raw=False, return_unc=True, **kwargs):
        self.model = 0
        self.model_class = 0
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.model_kwargs = kwargs
        self.return_raw = return_raw
        self.return_unc = return_unc
        if method == "dropout":
            self.model_class = BNN_Dropout

        #task modes
        if type(task) == set:
            task = "classification"
        self.task = task
        if task == "classification":
            criterion = torch.nn.CrossEntropyLoss()
        elif task == "regression":
            criterion = torch.nn.MSELoss()
        self.criterion = criterion

    def fit(self, x, y=None):

        #prepare model
        if self.task == "classification" and y is not None:
            output_size = np.unique(y).shape[0]
        else:
            output_size = 1

        if "Tensor" not in type(x).__name__:
            x = Variable(torch.from_numpy(x).float())
        if y is None:
            y = x
        if "Tensor" not in type(y).__name__:
            if self.task == "classification":
                y = Variable(torch.from_numpy(y).long())
            else:
                y = Variable(torch.from_numpy(y).float())

        self.model = self.model_class(input_size=x.shape[-1],output_size=output_size,**self.model_kwargs)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        #train for n epochs
        for epoch in range(self.num_epochs):
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self

    def predict(self, x):
        """
        By default, returns a tuple of (y_pred,y_unc) corresponding to the best estimate and uncertainty
        Has option to return raw predictions samples from the model has the following dimensions: (num_samples, num_examples, output_size)

        Parameters
        ----------

        return_raw: bool
            If true, return raw samples to be processed by user.
        return_unc: bool
            If true, return uncertainty along with best estimates

        """

        if "Tensor" not in type(x).__name__:
            x = Variable(torch.from_numpy(x).float())
        y_pred_raw = np.array([])
        #if classification, apply log softmax to output
        if self.task == "classification":
            y_pred_raw = np.array([torch.nn.functional.log_softmax(self.model(x),dim=1).detach().cpu().numpy() for i in range(self.num_samples)])
            y_pred_raw =torch.topk(torch.from_numpy(y_pred_raw).float(),k=1,dim=-1)[1].squeeze()
            y_pred_raw = y_pred_raw.detach().cpu().numpy()
        else:
            y_pred_raw = np.array([self.model(x).detach().cpu().numpy() for i in range(self.num_samples)])
        if self.return_raw:
            return y_pred_raw

        if self.task == "classification":
            #best estimate
            y_pred = stats.mode(y_pred_raw)[0].reshape(-1)

            #calculate uncertainty
            if self.return_unc == True:
                y_unc = np.random.randn(y_pred.shape[0])
                for example in range(y_pred_raw.shape[1]):
                    unique, counts = np.unique(y_pred_raw[:,example], return_counts=True)
                    y_unc[example] = float(counts.max()/counts.sum())
                return y_pred,y_unc
            else:
                return y_pred
        elif self.task == "regression":
            y_pred = np.mean(y_pred_raw,axis=0).squeeze()
            y_unc = np.std(y_pred_raw,axis=0).squeeze()
            if self.return_unc == True:
                return y_pred, y_unc
            else:
                return y_pred
        return -1
