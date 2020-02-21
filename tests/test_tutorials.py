from agentMET4FOF_tutorials.tutorial_1_generator_agent import main as tutorial_1_main
from agentMET4FOF_tutorials.tutorial_2_math_agent import main as tutorial_2_main
from agentMET4FOF_tutorials.tutorial_3_multi_channel import main as tutorial_3_main


def test_tutorial_1():
    tutorial_1_main().shutdown()


def test_tutorial_2():
    tutorial_2_main().shutdown()


def test_tutorial_3():
    tutorial_3_main().shutdown()
