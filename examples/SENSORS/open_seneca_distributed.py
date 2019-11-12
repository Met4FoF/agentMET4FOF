from agentMET4FOF.agents import AgentMET4FOF, AgentNetwork, MonitorAgent
from agentMET4FOF.streams import SineGenerator

import serial
#run this on terminal to gain permission to the serial port for reading from sensor
#sudo chmod 666 /dev/ttyS0

#This example setups two computers for receiving and sending data from OpenSeneca sensor board
#1) Record the IP addresses of the first and second PC and saved as `host_ip` and `local_ip` variables below
#2) Setup a PC with an AgentNetwork server, such as running `main_agent_network.py` or any of the other tutorials
#by passing the argument `ip_addr=host_ip` when starting the AgentNetwork
#3) Connect the sensor board to the second PC (such as Raspberry Pi) and execute this code to start an agent
#on the second PC and subsequently connect to the first PC (host)
#Note: Firewall permission will need to be allowed on several ports : 3333 (agent server) and 8050 (web app)

host_ip='192.168.43.95'
local_ip='192.168.43.102'

class OpenSenecaAgent(AgentMET4FOF):
    def init_parameters(self, port_name='/dev/ttyUSB0', sensor_buffer_size=5):
        self.stream = self.connect_seneca_serial(port_name)
        self.buffer_size = sensor_buffer_size

    def agent_loop(self):
        if self.current_state == "Running":
            sensor_data = self.read_seneca_sensor(self.stream)

            #save data into memory
            self.update_data_memory({'from':self.name,'data':sensor_data})
            # send out buffered data if the stored data has exceeded the buffer size
            if len(self.memory[self.name][next(iter(self.memory[self.name]))]) >= self.buffer_size:
                self.send_output(self.memory[self.name])
                self.memory = {}

    def connect_seneca_serial(self, port_name = '/dev/ttyUSB0'):
        ser = serial.Serial(port_name)
        return ser

    def read_seneca_sensor(self, ser):
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
    #connect to agent network server
    log_file = False
    agentNetwork = AgentNetwork(dashboard_modules=False,log_filename=log_file, ip_addr=host_ip)

    #init agents by adding into the agent network
    gen_agent = agentNetwork.add_agent(agentType = OpenSenecaAgent,ip_addr=local_ip)
    gen_agent.init_parameters(port_name='/dev/ttyUSB0',sensor_buffer_size=1)
    gen_agent.init_agent_loop(loop_wait=1)

    # set all agents states to "Running"
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork

if __name__ == '__main__':
    main()

