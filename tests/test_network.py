import pytest

from agentMET4FOF.agents import AgentNetwork
from tests.conftest import test_timeout


@pytest.mark.timeout(test_timeout)
@pytest.mark.parametrize("backend", ["mesa", "osbrain"])
def test_shutdown(backend):
    # Check if the agent network gets properly setup and stopped again. The tear down
    # does not terminate cleanly for the "Mesa" backend yet, so we only test "osbrain".
    network = AgentNetwork(backend=backend)
    network.shutdown()
