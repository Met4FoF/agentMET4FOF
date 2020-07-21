from agentMET4FOF_tutorials.tutorial_1_generator_agent import (
    demonstrate_generator_agent_use as tut1,
)
from agentMET4FOF_tutorials.tutorial_2_math_agent import main as tut2
from agentMET4FOF_tutorials.tutorial_3_multi_channel import main as tut3
from agentMET4FOF_tutorials.tutorial_4_metrological_agents import main as tut4


def test_tutorial_1():
    # Test executability of tutorial_1_generator_agent.
    tut1().shutdown()


def test_tutorial_2():
    # Test executability of tutorial_2_math_agent.
    tut2().shutdown()


def test_tutorial_3():
    # Test executability of tutorial_3_multi_channel.
    tut3().shutdown()


def test_tutorial_4():
    # Test executability of tutorial_4_metrological_agents.py.
    tut4().shutdown()
