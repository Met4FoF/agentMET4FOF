import random
import string

import hypothesis.strategies as hst
from hypothesis import given
from hypothesis.strategies import characters

from agentMET4FOF.agents import AgentMET4FOF
from agentMET4FOF.network import AgentNetwork


@given(
    hst.text(
        characters(min_codepoint=1, max_codepoint=1000),
    )
)
def test_naming_agents_for_mesa(name):
    agent_network = AgentNetwork(dashboard_modules=False, backend="mesa")

    agent = agent_network.add_agent(name=name, agentType=AgentMET4FOF)
    assert agent_network.get_agent(name).name == agent.name == name

    agent_network.shutdown()


def test_naming_agents_for_osbrain(agent_network):
    random_name = "".join(
        random.choices(
            string.digits + string.ascii_letters + " ",
            k=random.randint(1, 100),
        )
    )
    agent = agent_network.add_agent(name=random_name, agentType=AgentMET4FOF)
    assert (
        agent_network.get_agent(random_name).get_attr("name")
        == agent.get_attr("name")
        == random_name.replace(" ", "_")
    )
