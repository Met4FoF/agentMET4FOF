import numpy as np
import datetime
from AgentMET4FOF import AgentMET4FOF, AgentNetwork, MonitorAgent, ML_Model, DataStream

from skmultiflow.data import SineGenerator
import plotly.offline as py
import plotly.tools as tls

import matplotlib.pyplot as plt
import numpy as np

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

def divide_by_two(data):
    return data / 2

class MathAgent(AgentMET4FOF):
    def on_received_message(self, message):
        data = divide_by_two(message['data'])
        self.send_output(data)

class GeneratorAgent(AgentMET4FOF):

    def create_graph(self):
        # Fixing random state for reproducibility
        np.random.seed(19680801)

        dt = 0.01
        t = np.arange(0, 30, dt)
        nse1 = np.random.randn(len(t))                 # white noise 1
        nse2 = np.random.randn(len(t))                 # white noise 2

        # Two signals with a coherent part at 10Hz and a random part
        s1 = np.sin(2 * np.pi * 10 * t) + nse1
        s2 = np.sin(2 * np.pi * 10 * t) + nse2

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(t, s1, t, s2)
        axs[0].set_xlim(0, 2)
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('s1 and s2')
        axs[0].grid(True)

        cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
        axs[1].set_ylabel('coherence')
        #fig.suptitle('Monitor Agent 1', fontsize=16)
        return fig

if __name__ == '__main__':
    # start agent network server
    agentNetwork = AgentNetwork()

    # init agents
    gen_agent = agentNetwork.add_agent(agentType=GeneratorAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)
    math_agent = agentNetwork.add_agent(agentType=MathAgent)

    agentNetwork.bind_agents(gen_agent, monitor_agent)

    gen_agent.send_plot()

    #monitor_agent.get_attr('plots') ==




