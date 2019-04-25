import osbrain
from osbrain.agent import run_agent
from osbrain.agent import Agent

import pandas as pd
from datetime import datetime

import time
# TYPES OF AGENT
# 0 - SENSOR NETWORK
# 1 - SENSOR
# 2 - AGGREGATOR

class Sensor(Agent):

    def on_init(self):
        self.log_info('SENSOR INITIALIZED')
        self.current_data =0
    def read_request(self,message):
        self.log_info('RECEIVED JOB: {}'.format(message) )
        self.send_data(message)
    def send_data(self,message=0):
        data = self.read_generator()
        self.log_info('SENDING OFF DATA : {}'.format(data))       
        self.send(self.sens_agg_PUSH, {self.name:data})
    def set_generator(self, generator_function):
        self.generator = generator_function
    def read_generator(self):
        return self.generator((datetime.now()-self.unit_t).microseconds)
    def read_data(self, message=0):
        #data source
        data = self.data_source[message,:]

        self.current_data = data
        self.log_info('Read Data: {}'.format(data))
        return data

class Aggregator(Agent):

    def on_init(self, timeout = 5):
        self.buffer ={}
        self.buffer_pd = pd.DataFrame(self.buffer)
        self.num_requests = 0
        self.num_sensors =0
        self.sensor_list = []
    def bind_sensors(self,sensor_list=[]):
        #SETUP AGENT-COMM FOR SENSOR-AGGREGATOR
        addr_PUB = self.bind('PUB', alias='PUB_SENS_AGG_1')
        self.sens_agg_PUB = addr_PUB

        for i in range(len(sensor_list)):
            sensor_list[i].connect(addr_PUB, handler=Sensor.read_request)
            addr_PULL = self.bind('PULL', alias='PUSH_SENS_AGG_'+str(i+1),handler=Aggregator.aggregate_sensor_data)
            addr_PUSH = sensor_list[i].bind('PUSH', alias='PUSH_SENS_AGG_'+str(i+1))
            self.connect(addr_PUSH, handler=Aggregator.aggregate_sensor_data)
            sensor_list[i].set_attr(sens_agg_PUSH = addr_PUSH)
        self.binded_sensors = sensor_list
        self.num_sensors = len(sensor_list)
        self.log_info("Binded all sensors")

    def aggregate_sensor_data(self, message):
        self.buffer.update(message)
        self.buffer_pd = pd.DataFrame(self.buffer)
        if self.check_fill_buffer():
            self.log_info("Received all data from Sensor Agents, ready to be sent off:")
            self.log_info("Buffer Data: " + str(self.buffer_pd))

    def request_sensors_data(self):
        self.num_requests = self.num_requests+1
        self.send(self.sens_agg_PUB, "Request #"+str(self.num_requests))
        self.log_info("Requesting data from Sensor Agents ")
    def clear_buffer(self):
        self.buffer = {}
        self.buffer_pd = pd.DataFrame(self.buffer)

    def check_fill_buffer(self,msg= None):
        if len(self.buffer) >= self.num_sensors :
            return True
        return False
    def get_buffer_data(self):
        return self.buffer_pd

class SensorNetwork(Agent):
    def on_init(self):
        self.sensor_list =[]
        self.aggregator_list = []
    def get_numSensors(self):
        return len(self.sensor_list)
    def get_numAggregators(self):
        return len(self.aggregator_list)

    def add_simsensor(self, type="force", unit_v = "N", unit_t=datetime.now(), id=" ", generator=""):
        #if sensor_id is not provided by user, then resort to generic names
        if id == " " :
            sensor_id = 'sensor_' +type+"_"+ str(self.get_numSensors())
        else:
            sensor_id = id
        new_sensor = run_agent(sensor_id, base=Sensor)
        new_sensor.set_attr(type=type, unit_v = unit_v, unit_t=unit_t, id=sensor_id)
        new_sensor.set_generator(generator)
        self.sensor_list.append(new_sensor)
        self.log_info("sensor added generator function")
        return new_sensor

    def add_aggregator(self,sensor_list=[]):
        new_aggregator = run_agent('aggregator_' + str(self.get_numAggregators()), base=Aggregator)
        self.aggregator_list.append(new_aggregator)
        new_aggregator.bind_sensors(sensor_list)
        return new_aggregator


def gen1(t=0):
    x = 0.3*t-1
    unc_x = 0.01
    return x, unc_x
def gen2(t=0):
    x = 10*t-5
    unc_x = 0.05
    return x, unc_x

if __name__ == '__main__':
   #SYSTEM INITIALIZE
   ns = osbrain.nameserver.run_nameserver(addr='127.0.0.1:14065')

   sensor_network = run_agent('sensor_network', base=SensorNetwork)

   #add sensors
   sensor1 = sensor_network.add_simsensor(generator=gen1, type="force", unit_v = "N", id='sensor_force1')
   sensor2 = sensor_network.add_simsensor(generator=gen2, type="temperature", unit_v = "F", id='sensor_temp1')
   sensor3 = sensor_network.add_simsensor(generator=gen2, type="speed", unit_v = "ms-1", id='sensor_speed1')
   sensor4 = sensor_network.add_simsensor(generator=gen1, type="force", unit_v = "N", id='sensor_force2')

   #access sensors by either way
   sensor_network.get_attr('sensor_list')[0].read_generator()
   sensor1.read_generator()

   #add aggregators
   aggregator1 = sensor_network.add_aggregator([sensor1,sensor2])
   aggregator2 = sensor_network.add_aggregator([sensor3,sensor4])

   #send request to aggregator agents for data from sensors
   aggregator1.request_sensors_data()
   aggregator2.request_sensors_data()

   #wait for aggregator buffer to be filled and store data in variable
   while aggregator1.check_fill_buffer() == False:
       time.sleep(1)
   data_requested = aggregator1.get_buffer_data()
   aggregator1.clear_buffer()

   #print sensor info
   print(sensor1.get_attr('type'),sensor1.get_attr('unit_v'),sensor1.get_attr('id'))

   #print requested data
   print(data_requested)

   #list of agents in system
   print(ns.agents())
  
   ns.shutdown()
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   