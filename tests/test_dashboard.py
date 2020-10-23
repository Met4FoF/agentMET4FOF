from multiprocessing.context import Process

import psutil
import pytest
from requests import head
from requests.exceptions import ConnectionError

from agentMET4FOF_tutorials.tutorial_1_generator_agent import (
    demonstrate_generator_agent_use,
)
from tests.conftest import test_timeout


@pytest.fixture()
def dashboard():
    # This fixture guarantees the proper termination of all spawned subprocesses
    # after the tests.
    dashboard = Process(target=demonstrate_generator_agent_use)
    dashboard.start()
    yield
    for child in psutil.Process(dashboard.pid).children(recursive=True):
        child.kill()
    dashboard.terminate()
    dashboard.join()


@pytest.mark.timeout(test_timeout)
@pytest.mark.usefixtures("dashboard")
def test_demo_dashboard():
    # This test calls demonstrate_generator_agent_use and waits for five seconds for the
    # process to bring up the Dashboard. If that did not happen the test times out
    # and thus fails.
    is_down = True
    while is_down:
        try:
            is_down = head("http://127.0.0.1:8050").status_code != 200
        except ConnectionError:
            pass
