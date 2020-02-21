from multiprocessing.context import Process

import pytest
from requests import head
from requests.exceptions import ConnectionError

from examples.ML.coupled_ML import main as coupled_ml_main
from examples.ML.decoupled_ML import main as decoupled_ml_main
from examples.ZEMA_EMC.main_zema_agents import main as zema_main
from examples.demo.demo_agent_network import main as demo_agent_network_main
from examples.demo.demo_agents import main as demo_agents_main
from examples.demo.run_dashboard import run_dashboard as demo_dashboard_main


class TestDemo:
    def test_demo_agent_network(self):
        demo_agent_network_main().shutdown()

    def test_demo_agents(self):
        demo_agents_main().shutdown()

    @pytest.mark.timeout(1)
    def test_demo_dashboard(self):
        # This test runs the run() method of demo_dashboard_main and waits for the
        # process to bring up the Dashboard. If that did not happen in a second,
        # the test times out.
        dashboard = Process(target=demo_dashboard_main)
        dashboard.start()
        is_down = True
        while is_down:
            try:
                is_down = head("http://127.0.0.1:8050").status_code != 200
            except ConnectionError:
                pass
        dashboard.terminate()
        dashboard.join()


class TestML:
    def test_coupled_ML(self):
        coupled_ml_main().shutdown()

    def test_decoupled_ML(self):
        decoupled_ml_main().shutdown()


class TestZEMA_EMC:
    def test_main_zema_agents(self):
        zema_main().shutdown()
