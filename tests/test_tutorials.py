from agentMET4FOF_tutorials.tutorial_1_generator_agent import (
    demonstrate_generator_agent_use as tut1,
)
from agentMET4FOF_tutorials.tutorial_2_math_agent import main as tut2
from agentMET4FOF_tutorials.tutorial_3_multi_channel import main as tut3
from agentMET4FOF_tutorials.tutorial_4_metrological_streams import (
    demonstrate_metrological_stream as tut4,
)
from agentMET4FOF_tutorials.tutorial_5_coalition import (
    demonstrate_generator_agent_use as tut5,
)
from agentMET4FOF_tutorials.tutorial_6_mesa_backend import (
    demonstrate_mesa_backend as tut6,
)
from agentMET4FOF_tutorials.tutorial_7_generic_metrological_agent import (
    demonstrate_metrological_stream as tut7,
)


def test_tutorial_1():
    """Test executability of tutorial_1_generator_agent."""
    tut1().shutdown()


def test_tutorial_2():
    """Test executability of tutorial_2_math_agent."""
    tut2().shutdown()


def test_tutorial_3():
    """Test executability of tutorial_3_multi_channel."""
    tut3().shutdown()


def test_tutorial_4():
    """Test executability of tutorial_4_metrological_streams.py."""
    tut4().shutdown()


def test_tutorial_5():
    """Test executability of tutorial_5_coalition.py."""
    tut5().shutdown()


def test_tutorial_6():
    """Test executability of tutorial_6_mesa_backend.py."""
    tut6().shutdown()


def test_tutorial_7():
    """Test executability of tutorial_7_generic_metrological_agent.py."""
    tut7().shutdown()
