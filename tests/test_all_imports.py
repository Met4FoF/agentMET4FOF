def test_legacy_imports():
    from agentMET4FOF import agents, streams
    from agentMET4FOF.agents import (
        AgentBuffer,
        AgentMET4FOF,
        DataStreamAgent,
        MonitorAgent,
        SineGeneratorAgent,
    )
    from agentMET4FOF.network import AgentNetwork
    from agentMET4FOF.streams import DataStreamMET4FOF, SineGenerator, CosineGenerator


def test_metrological_base_agents_imports():
    from agentMET4FOF.agents.metrological_base_agents import (
        MetrologicalAgent,
        MetrologicalMonitorAgent,
    )


def test_metrological_signal_agents_import():
    from agentMET4FOF.agents.metrological_signal_agents import (
        MetrologicalGeneratorAgent,
    )


def test_signal_agents_import():
    from agentMET4FOF.agents.signal_agents import (
        NoiseAgent,
        SineGeneratorAgent,
        SineWithJitterGeneratorAgent,
        StaticSineWithJitterGeneratorAgent,
    )


def test_noise_jitter_removal_agents_import():
    from agentMET4FOF.agents.noise_jitter_removal_agents import (
        NoiseJitterRemovalAgent,
        MCMCMHNJ,
    )


def test_agents_buffer_imports():
    from agentMET4FOF.utils.buffer import (
        AgentBuffer,
        MetrologicalAgentBuffer,
    )


def test_agents_network_import():
    from agentMET4FOF.network import (
        AgentNetwork,
    )


def test_base_agents_imports():
    from agentMET4FOF.agents.base_agents import (
        AgentMET4FOF,
        DataStreamAgent,
        MonitorAgent,
    )


def test_metrological_redundancy_agents_imports():
    from agentMET4FOF.agents.metrological_redundancy_agents import (
        RedundancyAgent,
    )


def test_base_streams_import():
    from agentMET4FOF.streams.base_streams import (
        DataStreamMET4FOF,
    )


def test_metrological_base_streams_import():
    from agentMET4FOF.streams.metrological_base_streams import (
        MetrologicalDataStreamMET4FOF,
    )


def test_signal_streams_import():
    from agentMET4FOF.streams.signal_streams import (
        SineGenerator,
        CosineGenerator,
        SineWithJitterGenerator,
        StaticSineWithJitterGenerator,
    )


def test_metrological_signal_streams_import():
    from agentMET4FOF.streams.metrological_signal_streams import (
        MetrologicalMultiWaveGenerator,
        MetrologicalSineGenerator,
    )


def test_dashboard_import():
    from agentMET4FOF.dashboard import (
        Dashboard,
        Dashboard_agt_net,
        Dashboard_Control,
        Dashboard_layout_base,
        default_network_stylesheet,
        LayoutHelper,
    )


def test_tutorials_import():
    from agentMET4FOF_tutorials import (
        tutorial_1_generator_agent,
        tutorial_2_math_agent,
        tutorial_3_multi_channel,
        tutorial_4_metrological_streams,
        tutorial_5_coalition,
        tutorial_6_mesa_backend,
        tutorial_7_generic_metrological_agent,
    )


def test_utils_import():
    from agentMET4FOF.utils import (
        AgentBuffer,
        Backend,
        MetrologicalAgentBuffer,
    )
