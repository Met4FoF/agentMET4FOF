from multiprocessing.context import Process

import pytest
from requests import head
from requests.exceptions import ConnectionError

from agentMET4FOF_tutorials.tutorial_1_generator_agent import (
    demonstrate_generator_agent_use,
)


@pytest.mark.timeout(5)
def test_demo_dashboard():
    # This test runs the run() method of demo_dashboard_main and waits for the
    # process to bring up the Dashboard. If that did not happen in a second,
    # the test times out.
    dashboard = Process(target=demonstrate_generator_agent_use)
    dashboard.start()
    is_down = True
    while is_down:
        try:
            is_down = head("http://127.0.0.1:8050").status_code != 200
        except ConnectionError:
            pass
    dashboard.terminate()
    dashboard.join()
