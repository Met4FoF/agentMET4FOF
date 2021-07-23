import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from scipy.stats import multivariate_normal as mvn

from agentMET4FOF.agents.metrological_redundancy_agents import RedundancyAgent


@pytest.fixture(scope="module")
def a_arr2d():
    # Prepare a_arr2d for calling lcss
    a_arr2d = np.array([[1, 2, 3, 4], [2, -5, 4, 1], [2, 9, 1, 0], [3, 5, -2, 4]])
    s = np.sum(a_arr2d, 1)
    s.shape = (s.shape[0], 1)  # set the second dimension to 1
    a_arr2d = a_arr2d / s  # make all row sums equal to 1
    return a_arr2d


@pytest.fixture(scope="module")
def x_arr():
    # Prepare x_arr for calling lcss
    return np.array([22.3, 20.6, 25.5, 19.3])


@pytest.fixture(scope="module")
def vx_arr2d():
    # Prepare vx_arr2d for calling lcss
    return np.identity(4) + np.ones((4, 4))


@pytest.fixture(scope="module")
def a_arr():
    # Prepare a_arr for calling lcss
    return np.zeros(4)


@pytest.fixture(scope="module")
def problim():
    # Prepare problim for all those tests
    return 0.95


@pytest.fixture(scope="module")
def compare_lcss_to_expected():
    def comparison(a_arr, a_arr2d, x_arr, vx_arr2d, problim):
        redagent = RedundancyAgent()
        n_sols, ybest, uybest, chi2obs, indkeep = redagent.calc_lcss(
            a_arr, a_arr2d, x_arr, vx_arr2d, problim
        )

        assert n_sols == 1
        assert_almost_equal(ybest, 20.733333333333334)
        assert_almost_equal(uybest, 1.1547005383792515)
        assert_almost_equal(chi2obs, 4.5266666666666655)
        assert_array_equal(indkeep, np.array([0, 1, 3]))

    return comparison


def test_calc_consistent_estimates_no_corr_with_one_element_example():
    y_arr = np.array([20.2, 21.3, 20.5])
    uy_arr = np.array([0.5, 0.8, 0.3])
    prob_lim = 0.05

    (
        isconsist,
        ybest,
        uybest,
        chi2obs,
    ) = RedundancyAgent.calc_consistent_estimates_no_corr(y_arr, uy_arr, prob_lim)

    assert isconsist == [True]
    assert_almost_equal(ybest, 20.502998750520614)
    assert_almost_equal(uybest, 0.24489795918367346)
    assert_almost_equal(chi2obs[0], 1.35985006)


def test_calc_consistent_estimates_no_corr_with_two_element_example():
    y_arr = np.array([[20.2, 21.3, 20.5], [19.5, 19.7, 20.3]])
    uy_arr = np.array([[0.5, 0.8, 0.3], [0.1, 0.2, 0.3]])
    prob_lim = 0.05

    (
        isconsist_arr,
        ybest_arr,
        uybest_arr,
        chi2obs_arr,
    ) = RedundancyAgent.calc_consistent_estimates_no_corr(y_arr, uy_arr, prob_lim)

    assert_array_equal(isconsist_arr, np.array([True, False]))
    assert_array_almost_equal(ybest_arr, np.array([[20.50299875], [19.60204082]]))
    assert_array_almost_equal(uybest_arr, np.array([0.24489796, 0.08571429]))
    assert_array_almost_equal(chi2obs_arr, np.array([1.35985006, 6.69387755]))


def test_calc_best_estimate(problim):
    y_arr = np.array([20.2, 20.5, 20.8])
    vy_arr2d = np.array([[2, 1, 1], [1, 3, 1], [1, 1, 4]])
    isconsist, ybest, uybest, chi2obs = RedundancyAgent.calc_best_estimate(
        y_arr, vy_arr2d, problim
    )
    assert isconsist
    assert_almost_equal(ybest, 20.39090909090909)
    assert_almost_equal(uybest, 1.2431631210161223)
    assert_almost_equal(chi2obs, 0.09818181818181865)


def test_calc_best_estimate_limprob(problim):
    n_reps = 10000
    ymean = 20.0
    vy_arr2d = np.random.rand(4, 4)
    vy_arr2d = vy_arr2d.transpose() @ vy_arr2d
    n_casekeep = 0
    for i_rep in range(n_reps):
        y_arr = ymean + mvn.rvs(mean=None, cov=vy_arr2d)
        isconsist, ybest, uybest, chi2obs = RedundancyAgent.calc_best_estimate(
            y_arr, vy_arr2d, problim
        )
        if isconsist:
            n_casekeep += 1
    frackeep = n_casekeep / n_reps
    assert_almost_equal(frackeep, problim, decimal=3)


def test_calc_lcs_one_solution(problim):
    y_arr = np.array([20, 20.6, 20.5, 19.3])
    vy_arr2d = np.identity(4) + np.ones((4, 4))
    redagent = RedundancyAgent()
    n_sols, ybest, uybest, chi2obs, indkeep = redagent.calc_lcs(
        y_arr, vy_arr2d, problim
    )

    assert n_sols == 1
    assert_almost_equal(ybest, 20.1)
    assert_almost_equal(uybest, 1.118033988749895)
    assert_almost_equal(chi2obs, 1.0600000000000003)
    assert_array_equal(indkeep, np.array([0, 1, 2, 3]))


def test_calc_lcs_two_solutions(problim):
    y_arr = np.array([10, 11, 20, 21])
    vy_arr2d = 5 * np.identity(4) + np.ones((4, 4))
    redagent = RedundancyAgent()

    n_sols, ybest, uybest, chi2obs, indkeep = redagent.calc_lcs(
        y_arr, vy_arr2d, problim
    )

    assert n_sols == 2
    assert_array_almost_equal(ybest, np.array([10.5, 20.5]))
    assert_array_almost_equal(uybest, np.array([1.87082869, 1.87082869]))
    assert_almost_equal(chi2obs, 0.1)
    assert_array_equal(indkeep, np.array([[0.0, 1.0], [2.0, 3.0]]))


def test_calc_lcs_limprob(problim):
    n_reps = 10000
    redagent = RedundancyAgent()
    ymean = 20.0
    vy_arr2d = np.random.rand(4, 4)
    vy_arr2d = vy_arr2d.transpose() @ vy_arr2d
    n_casekeep = 0
    for i_rep in range(n_reps):
        y_arr = ymean + mvn.rvs(mean=None, cov=vy_arr2d)
        n_sols, ybest, uybest, chi2obs, indkeep = redagent.calc_lcs(
            y_arr, vy_arr2d, problim
        )
        if indkeep.shape[-1] == len(y_arr):
            n_casekeep += 1
    frackeep = n_casekeep / n_reps
    assert_almost_equal(frackeep, problim, decimal=2)


def test_calc_lcss_same_as_lcs(
    x_arr, vx_arr2d, a_arr, problim, compare_lcss_to_expected
):
    a_arr2d = np.identity(4)

    compare_lcss_to_expected(a_arr, a_arr2d, x_arr, vx_arr2d, problim)


def test_lcss_to_lcs_reduction(a_arr2d, problim, compare_lcss_to_expected):
    # Manipulate input to create a non trivial vector a_arr
    x_arr = np.array([22.3, 20.6, 25.5, 19.3])
    vx_arr2d = np.identity(4) + np.ones((4, 4))
    a_arr = np.zeros(4)
    dx_arr = np.array([1, 2, 3, 4])
    x_arr = x_arr - dx_arr
    a_arr = a_arr + np.matmul(a_arr2d, dx_arr)

    compare_lcss_to_expected(a_arr, a_arr2d, x_arr, vx_arr2d, problim)


def test_calc_lcss_two_solutions(problim):
    x_arr = np.array([10, 11, 20, 21])
    vx_arr2d = np.identity(4) + np.ones((4, 4))
    a_arr2d = np.array([[1, 2, 3, 4], [2, -5, 4, 1], [2, 9, 1, 0], [3, 5, -2, 4]])

    # Manipulate input to create a non trivial vector a_arr
    a_arr = np.zeros(4)
    dx_arr = np.array([1, 20, 3, -44])
    x_arr = x_arr - dx_arr
    a_arr = a_arr + np.matmul(a_arr2d, dx_arr)

    redagent = RedundancyAgent()
    n_sols, ybest, uybest, chi2obs, indkeep = redagent.calc_lcss(
        a_arr, a_arr2d, x_arr, vx_arr2d, problim
    )

    assert n_sols == 1
    assert_almost_equal(ybest, 337.67088607594957)
    assert_almost_equal(uybest, 22.73234465845718)
    assert_almost_equal(chi2obs, 0.01928872814948826)
    assert_array_equal(indkeep, np.array([1, 3]))


def test_calc_lcss_limprob(a_arr2d, problim):
    n_reps = 1000
    xmean = 20.0
    vx_arr2d = np.random.rand(4, 4)
    vx_arr2d = vx_arr2d.transpose() @ vx_arr2d
    n_casekeep = 0
    redagent = RedundancyAgent()

    for i_rep in range(n_reps):
        x_arr = xmean + mvn.rvs(mean=None, cov=vx_arr2d)
        # Add an additional conversion to work with non-trivial vector a_arr
        dx_arr = np.random.standard_normal(4)
        x_arr = x_arr - dx_arr
        a_arr = np.matmul(a_arr2d, dx_arr)
        n_sols, ybest, uybest, chi2obs, indkeep = redagent.calc_lcss(
            a_arr, a_arr2d, x_arr, vx_arr2d, problim
        )
        if indkeep.shape[-1] == len(x_arr):
            n_casekeep += 1
    frackeep = n_casekeep / n_reps
    assert_almost_equal(frackeep, problim, decimal=2)
