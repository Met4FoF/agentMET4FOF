from AgentMET4FOF import AgentMET4FOF

def minus(data, minus_val):
    return data-minus_val

def plus(data,plus_val):
    return data+plus_val

class SubtractAgent(AgentMET4FOF):
    def init_parameters(self,minus_param=0.5):
        self.minus_param = minus_param

    def on_received_message(self, message):
        minus_data = minus(message['data']['x'], self.minus_param)

        self.send_output(minus_data)

class AdditionAgent(AgentMET4FOF):
    def init_parameters(self,plus_param=0.5):
        self.plus_param = plus_param

    def on_received_message(self, message):
        plus_data = plus(message['data']['x'], self.plus_param)

        self.send_output(plus_data)
