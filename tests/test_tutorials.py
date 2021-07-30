from agentMET4FOF_tutorials.noise_jitter.generate_sine_with_jitter import (
    demonstrate_sine_with_jitter_agent_use,
)
from agentMET4FOF_tutorials.noise_jitter.add_noise_to_existing_signal import (
    demonstrate_noise_agent_use,
)
from agentMET4FOF_tutorials.noise_jitter.remove_noise_and_jitter import (
    demonstrate_noise_jitter_removal_agent,
)
from agentMET4FOF_tutorials.redundancy.redundancy_agent_four_signals import (
    demonstrate_redundancy_agent_four_signals as four_signal_redundancy,
)
from agentMET4FOF_tutorials.redundancy.redundancy_agent_one_signal import (
    demonstrate_redundancy_agent_onesignal as one_signal_redundancy,
)
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


def test_tutorial_1_executability():
    tut1().shutdown()


def test_tutorial_2_executability():
    tut2().shutdown()


def test_tutorial_3_executability():
    tut3().shutdown()


def test_tutorial_4_executability():
    tut4().shutdown()


def test_tutorial_5_executability():
    tut5().shutdown()


def test_tutorial_6_executability():
    tut6().shutdown()


def test_tutorial_7_executability():
    tut7().shutdown()


def test_redundancy_agent_four_signals_executability():
    four_signal_redundancy().shutdown()


def test_redundancy_agent_one_signal_executability():
    one_signal_redundancy().shutdown()


def test_noise_agent_demo_executability():
    demonstrate_noise_agent_use().shutdown()


def test_sine_with_jitter_demo_executability():
    demonstrate_sine_with_jitter_agent_use().shutdown()


def test_remove_noise_and_jitter_demo_executability():
    demonstrate_noise_jitter_removal_agent().shutdown()
