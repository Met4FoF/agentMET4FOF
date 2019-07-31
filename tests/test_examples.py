from examples.demo.demo_agent_network import main as demo_agent_network_main
from examples.demo.demo_agents import main as demo_agents_main
from examples.ML.coupled_ML import main as coupled_ml_main
from examples.ML.decoupled_ML import main as decoupled_ml_main
from examples.ZEMA_EMC.main_zema_agents import main as zema_main


class TestDemo:

    def test_demo_agent_network(self):
        demo_agent_network_main().shutdown()

    def test_demo_agents(self):
        demo_agents_main().shutdown()


class TestML:

    def test_coupled_ML(self):
        coupled_ml_main().shutdown()

    def test_decoupled_ML(self):
        decoupled_ml_main().shutdown()


class TestZEMA_EMC:

    def test_main_zema_agents(self):
        zema_main().shutdown()
