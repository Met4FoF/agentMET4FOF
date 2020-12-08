from agentMET4FOF import agents, streams
from agentMET4FOF.dashboard import (
    Dashboard,
    Dashboard_agt_net,
    Dashboard_Control,
)
from agentMET4FOF_tutorials import (
    tutorial_1_generator_agent,
    tutorial_2_math_agent,
    tutorial_3_multi_channel,
    tutorial_4_metrological_streams,
    tutorial_5_coalition,
)

# During a pytest execution this file gets interpreted once and all imports at the
# top are executed once.
#
# So this file just ensures that all imports would work as expected. This of course is
# just a minimal version of ensuring, that everything is there and basic dependencies
# are met as well. We should improve this and replace it by proper testing of all
# components of agentMET4FOF.
