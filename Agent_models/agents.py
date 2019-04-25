import osbrain
from osbrain.agent import run_agent
from osbrain.agent import Agent

import pandas as pd
from datetime import datetime

import time

import pickle
import numpy as np
import random
from copy import copy

class Sensor(Agent):
    def on_init(self):
        self.log_info('SENSOR INITIALIZED')
        self.data_index =0
        self.current_data = 0
        self.pushPull_commStreams = {"aggregator_id": []}

    def read_request(self, message):
        self.log_info('RECEIVED JOB: {}'.format(message))
        self.send_data(message)

    def send_data(self, message=0):
        self.current_data = self.read_generator()
        self.log_info('SENDING OFF DATA : {}'.format(self.current_data))
        self.send(self.pushPull_commStreams[message["agg_id"]], {self.name: self.current_data})

    def set_generator(self, generator_function):
        self.generator = generator_function

    def read_generator(self):
        return self.generator(t=(datetime.now() - self.unit_t).microseconds)

    def set_generatorDataSet(self, dataset):
        self.dataset = dataset

        def newGenerator(t=0):
            data_row = self.dataset[self.data_index,:]
            self.data_index = self.data_index+1
            if (self.data_index >= self.dataset.shape[0]):
                self.data_index =0
            return(data_row)
        self.generator = newGenerator




    def read_data(self, message=0):
        # data source
        data = self.data_source[message, :]

        self.current_data = data
        self.log_info('Read Data: {}'.format(data))
        return data


class Aggregator(Agent):
    def on_init(self, timeout=5):
        self.buffer = {}
        self.buffer_pd = pd.DataFrame(self.buffer)
        self.num_requests = 0
        self.num_sensors = 0
        self.sensor_list = []
        self.pred_agg_REP =""
        self.sens_agg_PUB =""
    def bind_sensors(self, sensor_list=[]):
        # SETUP AGENT-COMM FOR SENSOR-AGGREGATOR
        addr_PUB = self.bind('PUB', alias='PUB_SENS_AGG_'+self.name)
        self.sens_agg_PUB = addr_PUB

        for i in range(len(sensor_list)):
            # connect pub-sub (agg-sens)
            sensor_list[i].connect(addr_PUB, handler=Sensor.read_request)

            # connect push-pull (sens-agg)
            addr_PULL = self.bind('PULL', alias='PUSH_SENS_AGG_' + str(i + 1), handler=Aggregator.aggregate_sensor_data)
            addr_PUSH = sensor_list[i].bind('PUSH', alias='PUSH_SENS_AGG_' + str(i + 1))
            self.connect(addr_PUSH, handler=Aggregator.aggregate_sensor_data)
            temp_streams = sensor_list[i].get_attr("pushPull_commStreams")
            temp_streams[self.name] = addr_PUSH
            sensor_list[i].set_attr(pushPull_commStreams=temp_streams)

        self.binded_sensors = sensor_list
        self.num_sensors = len(sensor_list)
        self.log_info("Binded all sensors")

    def aggregate_sensor_data(self, message):
        self.buffer.update(message)
        self.buffer_pd = pd.DataFrame(self.buffer)
        if self.check_fill_buffer():
            pd_columns = [int(x.split('_')[-1]) for x in self.buffer_pd.columns]
            self.buffer_pd.columns = pd_columns
            print(self.buffer_pd.columns)
            self.buffer_pd.sort_index(axis=1, inplace=True)
            self.log_info("Received all data from Sensor Agents, ready to be sent off:")
            self.log_info("Buffer Data: " + str(self.buffer_pd))
            if self.pred_agg_REP != "":
                self.log_info(self.pred_agg_REP)
                self.send(self.pred_agg_REP,self.buffer_pd)
                #reply = self.recv(self.pred_agg_REP)
                #self.log_info(str(reply))
    def request_sensors_data(self):
        self.num_requests = self.num_requests + 1
        self.send(self.sens_agg_PUB, {"agg_id": self.name, "request_no": self.num_requests})
        self.log_info("Requesting data from Sensor Agents ")

    def clear_buffer(self):
        self.buffer = {}
        self.buffer_pd = pd.DataFrame(self.buffer)

    def check_fill_buffer(self, msg=None):
        if len(self.buffer) >= self.num_sensors:
            return True
        return False

    def get_buffer_data(self):
        return self.buffer_pd

class Predictor(Agent):
    def on_init(self,predictor_model=""):
        self.predictor_model = predictor_model
        self.pred_agg_REP =""
        self.current_prediction =0

    def load_predictor_model(self,predictor_model):
        self.predictor_model = predictor_model

    def process_aggregated_data(self, message):
        self.current_prediction = self.predict(message)  #now compute predict
        self.log_info("PREDICTION:"+str(self.current_prediction))
        self.send(self.deci_pred_REP, {self.name:{"pred":self.current_prediction[0][0],"unc":self.current_prediction[1][0]}} )

    def bind_aggregator(self,aggregator,mode="PUB"):
        if(mode =="PUSH"):
            addr_REP = aggregator.bind('PUSH', alias='PRED_AGG_'+self.name)
            self.connect(addr_REP, alias='PRED_AGG_'+self.name, handler=Predictor.process_aggregated_data)
            aggregator.set_attr(pred_agg_REP=addr_REP)
            self.log_info("Binded with Aggregator: Push-pull mode")
        elif(mode =="PUB"):
            if aggregator.get_attr("pred_agg_REP") == "":
                addr_PUB = aggregator.bind('PUB', alias='PRED_AGG_'+self.name)
                self.connect(addr_PUB, handler=Predictor.process_aggregated_data)
                aggregator.set_attr(pred_agg_REP=addr_PUB)
            else:
                addr_PUB = aggregator.get_attr("pred_agg_REP")
                self.connect(addr_PUB, handler=Predictor.process_aggregated_data)
            self.log_info("Binded with Aggregator: Pub-sub mode")
    def predict(self,data_input):
        if(self.predictor_model == ""):
            self.log_info("Error: No Model loaded")
            output ="NULL_PREDICTION"
            return output
        else:
            #print(data_input.values.shape)
            dt_temp = np.array([data_input.values])
            df_feats_test = self.predictor_model.extract_features(dt_temp)
            x_test = df_feats_test.values
            pred,unc,_ = self.predictor_model.predict_model_wUnc(x_test,num_samples=25)

            return (pred,unc)


class DecisionMaker(Agent):
    def on_init(self,reasoning_network=""):
        self.reasoning_network = reasoning_network
        self.buffer_inference ={}
        self.current_inference =[]
        self.binded_predictors =[]

    def load_reasoning_network(self,reasoning_network):
        self.reasoning_network = reasoning_network
    def infer(self,data_input,threshold = 80):
        certain = 1 if data_input['unc'] >=threshold else 0
        return certain
    def bind_predictor(self,predictor):
        addr_REP = predictor.bind('PUSH', alias='DEC_PRED_'+self.name)
        self.connect(addr_REP, alias='DEC_PRED_'+self.name, handler=DecisionMaker.process_prediction)
        predictor.set_attr(deci_pred_REP=addr_REP)
        self.log_info("Binded with Predictor")
        self.binded_predictors.append(predictor)
    def process_prediction(self,message):
        pred_agent_name = list(message)[0]
        uncertain_state = self.infer(message[pred_agent_name])
        message.update({pred_agent_name:{"unc_state":uncertain_state, "pred":message[pred_agent_name]['pred'],"unc":message[pred_agent_name]['unc']}})
        self.log_info(message)
        self.buffer_inference.update(message)
        if len(self.buffer_inference) >= len(self.binded_predictors):
            self.current_inference = pd.DataFrame(self.buffer_inference)
            self.buffer_inference = {}
            self.log_info(self.current_inference)

class SensorNetwork(Agent):
    def on_init(self):
        self.sensor_list = []
        self.aggregator_list = []
        self.predictor_list = []
        self.decisionMaker_list = []
    def get_numSensors(self):
        return len(self.sensor_list)

    def get_numAggregators(self):
        return len(self.aggregator_list)

    def get_numPredictors(self):
        return len(self.predictor_list)

    def get_numDecisionMaker(self):
        return len(self.decisionMaker_list)

    def add_simsensor(self, type="force", unit_v="N", unit_t=datetime.now(), id=" ", generator="", dataset=""):
        # if sensor_id is not provided by user, then resort to generic names
        if id == " ":
            sensor_id = 'sensor_' + type + "_" + str(self.get_numSensors())
        else:
            sensor_id = id
        new_sensor = run_agent(sensor_id, base=Sensor)
        new_sensor.set_attr(type=type, unit_v=unit_v, unit_t=unit_t, id=sensor_id)

        if(generator != ""):
            new_sensor.set_generator(generator)
        if(dataset != ""):
            new_sensor.set_generatorDataSet(dataset)

        self.sensor_list.append(new_sensor)
        self.log_info("sensor added generator function")
        return new_sensor

    def add_aggregator(self, sensor_list=[]):
        new_aggregator = run_agent('aggregator_' + str(self.get_numAggregators()), base=Aggregator)
        self.aggregator_list.append(new_aggregator)
        new_aggregator.bind_sensors(sensor_list)
        return new_aggregator

    def add_predictor(self, aggregator=""):
        new_predictor = run_agent('predictor_' + str(self.get_numPredictors()), base=Predictor)
        self.predictor_list.append(new_predictor)
        if aggregator !="":
            new_predictor.bind_aggregator(aggregator)
        return new_predictor

    def add_decisionMaker(self,predictor=""):
        new_decisionMaker = run_agent('decisionMaker_' + str(self.get_numDecisionMaker()), base=DecisionMaker)
        self.decisionMaker_list.append(new_decisionMaker)
        if predictor !="":
            new_decisionMaker.bind_predictor(predictor)
        return new_decisionMaker

