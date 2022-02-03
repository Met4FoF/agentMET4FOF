import pytest

from agentMET4FOF.network import AgentNetwork
from agentMET4FOF.utils import Backend
from tests.conftest import test_timeout


@pytest.mark.timeout(test_timeout)
@pytest.mark.parametrize("backend", [Backend.OSBRAIN, Backend.MESA])
def test_shutdown(backend):
    # Check if the agent network gets properly setup and stopped again.
    network = AgentNetwork(backend=backend)
    network.shutdown()
