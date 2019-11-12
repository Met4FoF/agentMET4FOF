import os
os.chdir("/home/pi/agentMET4FOF")

from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent

import time
import automationhat
import sys
import base64
import math
from io import BytesIO
from select import select
from pprint import pprint

sys.path.append("/home/pi/Robot/OctoprintPythonAPI")
from octoprint_api import Api

#We connect two types of sensors mounted on the 3D Printer to the Raspberry Pi here : Acceleration and Temperature
#The temperature sensor is built-in the printer and accessed by Octoprint API, while the acceleration sensor is installed
#and accessed via automationhat
#Two types of Sensor Agents are hence, built for these two main sensors. Further, the temperature can be set to
#be obtained from the separate positions if needed.
#These agents are connected to MonitorAgents to store and view the data


class AccelerationSensorAgent(AgentMET4FOF):
    def init_parameters(self, axis='x'):
        self.axis = axis
        return self
    def agent_loop(self):
        if self.current_state == "Running":
            sensor_data = self.next_sample() #dictionary
            self.send_output(sensor_data)

            #save data into memory
            self.update_data_memory({'from':self.name,'data':sensor_data})
            # send out buffered data if the stored data has exceeded the buffer size
            if len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size:
                self.send_output(self.memory[self.name])
                self.memory = {}

    def next_sample(self):
        data ={}
        if 'x' in self.axis:
            ADCX = automationhat.analog.one.read()
            accnX = 14 * (ADCX - 1.24) #accelerometer calibration
            data.update({'accnX':accnX})
        if 'y' in self.axis:
            ADCY = automationhat.analog.two.read()
            accnY = 14 * (ADCY - 1.22)
            data.update({'accnY':accnY})
        if 'z' in self.axis:
            ADCZ = automationhat.analog.three.read()
            accnZ = 14 * (ADCZ - 1.5)
            data.update({'accnZ':accnZ})
        return data


class TemperatureSensorAgent(AgentMET4FOF):
    """
    Several options are provided for the `position` selection : tool0, bed, or all to return both

    """
    def init_parameters(self, position='bed', base_url="http://192.168.100.2", api_key="44E9130C78A342F1837E5D706C478507"):
        printer = self.PrinterInitialise(base_url=base_url, api_key=api_key)
        self.printer = printer
        self.position = position
        return self

    def agent_loop(self):
        if self.current_state == "Running":
            sensor_data = self.next_sample() #dictionary
            # self.send_output(sensor_data)

            #save data into memory
            self.update_data_memory({'from':self.name,'data':sensor_data})
            # send out buffered data if the stored data has exceeded the buffer size
            if len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size:
                self.send_output(self.memory[self.name])
                self.memory = {}

    def next_sample(self):
        pState = self.printer.get_status()
        sensor_data = {}
        if self.position == 'all':
            for position in ['tool0','bed']:
                actual_temp = pState['temperature'][position]['actual']
                target_temp = pState['temperature'][position]['target']
                sensor_data.update({"actual_temp_"+position:actual_temp, "target_temp_"+position:target_temp})
        else:
            actual_temp = pState['temperature'][self.position]['actual']
            target_temp = pState['temperature'][self.position]['target']
            sensor_data = {"actual_temp_"+self.position:actual_temp, "target_temp_"+self.position:target_temp}
        return sensor_data

    def PrinterInitialise(self, base_url, api_key):
            printer = Api(base_url=base_url, api_key=api_key) #uses HTTP and JSON to communicate with 3D Printer software
            printer.disconnect()
            ConnectionState = printer.get_connection()
            printer.connect()
            while ConnectionState['current']['state'] != 'Operational':
                    sys.stdout.write('\n\r' + 'Waiting for printer connection...')
                    sys.stdout.flush()
                    ConnectionState = printer.get_connection()
                    #time.sleep(1)
            sys.stdout.write('\n')
            sys.stdout.flush()
            #pprint(ConnectionState)
            print ('Printer Connected')
            printer.home()
            time.sleep(10)
            print ('Printer homed')
            print (' ')
            return printer

def main():
    #start agent network server
    log_file = False
    agentNetwork = AgentNetwork(log_filename=log_file, dashboard_update_interval=1)

    #init agents by adding into the agent network
    #x_sensor_agent = agentNetwork.add_agent(name="X_acceleration",agentType= AccelerationSensorAgent).init_parameters(axis='y')
    #y_sensor_agent = agentNetwork.add_agent(name="Y_acceleration",agentType= AccelerationSensorAgent).init_parameters(axis='y')
    #z_sensor_agent = agentNetwork.add_agent(name="Z_acceleration",agentType= AccelerationSensorAgent).init_parameters(axis='z')
    sensor_buffer_size = 1
    acceleration_agent = agentNetwork.add_agent(name="XYZ_acceleration",agentType= AccelerationSensorAgent).init_parameters(axis='xyz',sensor_buffer_size=sensor_buffer_size)
    nozzle_sensor_agent = agentNetwork.add_agent(name="Nozzle_temperature",agentType= TemperatureSensorAgent).init_parameters(position='tool0',sensor_buffer_size=sensor_buffer_size)
    platform_sensor_agent = agentNetwork.add_agent(name="Platform_temperature",agentType= TemperatureSensorAgent).init_parameters(position='bed',sensor_buffer_size=sensor_buffer_size)

    sensor_agents = [acceleration_agent,nozzle_sensor_agent,platform_sensor_agent]

    #do something for all agents
    for sensor_agent in sensor_agents:
        #bind sensor agent to monitor
        monitor_agent = agentNetwork.add_agent(agentType= MonitorAgent, memory_buffer_size=100)
        sensor_agent.bind_output(monitor_agent)

        #set loop interval for each sensor agent
        sensor_agent.init_agent_loop(loop_wait=1)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
