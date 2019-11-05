from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator

import serial
import time
#run this on terminal to gain permission to the port
#sudo chmod 666 /dev/ttyS0

class OpenSinicaAgent(AgentMET4FOF):
    def init_parameters(self, port_name='/dev/ttyUSB0', sensor_buffer_size=5):
        self.stream = self.connect_sinica_serial(port_name)
        self.buffer_size = sensor_buffer_size

    def agent_loop(self):
        if self.current_state == "Running":
            sensor_data = self.read_sinica_sensor(self.stream)

            #save data into memory
            self.update_data_memory({'from':self.name,'data':sensor_data})
            # send out buffered data if the stored data has exceeded the buffer size
            if len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size:
                self.send_output(self.memory[self.name])
                self.memory = {}

    def connect_sinica_serial(self, port_name = '/dev/ttyUSB0'):
        ser = serial.Serial(port_name)
        return ser

    def read_sinica_sensor(self, ser):
        #setup headers
        headers = 'Counter,Latitude,Longitude,gpsUpdated,Speed,Altitude,Satellites,Date,Time,Millis,PM1.0,PM2.5,PM4.0,PM10,Temperature,Humidity,NC0.5,NC1.0,NC2.5,NC4.0,NC10,TypicalParticleSize,TVOC,eCO2,BatteryVIN'.split(',')
        num_headers = len(headers) #25

        #setup contents
        raw_sensor_read = ser.readline().decode("utf-8")[:-2]
        raw_sensor_read = raw_sensor_read.split(',')

        if num_headers == len(raw_sensor_read):
            packed_sensor_data = {header:raw_sensor_read[_id] for _id,header in enumerate(headers)}
            return packed_sensor_data
        return {}


def main():
    #start agent network server
    # log_mode = 'log_file.csv'
    log_mode = False
    agentNetwork = AgentNetwork(log_filename=log_mode)

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType = OpenSinicaAgent)
    monitor_agent = agentNetwork.add_agent(agentType = MonitorAgent, memory_buffer_size=10)
    gen_agent.init_parameters(port_name='/dev/ttyUSB0',sensor_buffer_size=3)

    #connect agents by either way:
    # 1) by agent network.bind_agents(source,target)
    agentNetwork.bind_agents(gen_agent, monitor_agent)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork



if __name__ == '__main__':
    main()

