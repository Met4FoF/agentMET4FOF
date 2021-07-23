from itertools import combinations
from typing import List

import numpy as np
from scipy.special import comb
from scipy.stats import chi2
from time_series_metadata.scheme import MetaData

from .metrological_base_agents import MetrologicalAgent
from ..exceptions import (
    ColumnNotZeroError,
    SensorsNotLinearlyIndependentError,
    SystemMatrixNotReducibleError,
)

__all__ = ["RedundancyAgent"]


class RedundancyAgent(MetrologicalAgent):
    """This is the main Redundancy Agent class

    `Redundancy` means that there is more than one way to derive the value of the
    measurand Y from the values of the sensor data X_i. Following main cases are
    considered in the agent:

    * Redundant measurement of the measurand Y by independent sensors directly
      measuring Y
    * Redundant measurement of the measurand Y by correlated sensors directly
      measuring Y
    * Redundant measurement of the measurand Y by correlated sensors X_i indirectly
      measuring Y, with a linear relationship y = a + A * x between the vector x of
      sensor values and the vector y containing the various (redundant) estimates of
      the measurand Y, where a is a vector and A a matrix both of appropriate size.

    Main calculations are performed in :py:func:`calc_lcs` and :py:func:`calc_lcss`.
    Usage of the :class:`RedundancyAgent` is relatively straightforward. Note that
    all static functions have `their own test functions
    <https://github.com/Met4FoF/agentMET4FOF/blob/develop/tests/
    test_redundancy_agent.py/>`_ illustrating their usage. Details of the different
    methods are presented in their respective docstrings.

    Please refer to other sections in this documentation for more information. A
    scientific publication explaining the ideas behind this agent can be found in
    [Kok2020_1]_. Related work can be found in [Kok2020_2]_.

    The usage of the Redundancy Agent is illustrated with two examples contained in
    :ref:`two tutorials <redundancy_tutorials>`.

    References
    ----------
    * Kok and Harris [Kok2020_1]_
    * Kok and Harris [Kok2020_2]_
    """

    metadata: MetaData
    calc_type: str
    sensor_key_list: List[str]
    n_pr: int
    problim: float
    a_arr: np.ndarray
    a_arr2d: np.ndarray

    def init_parameters(
        self,
        input_data_maxlen: int = 25,
        output_data_maxlen: int = 25,
        sensor_key_list: list = None,
        n_pr: int = 1,
        problim: float = 0.9,
        calc_type: str = "lcs",
    ):
        """Initialize the redundancy agent

        Parameters
        ----------
        input_data_maxlen : int, optional
            Defaults to 25
        output_data_maxlen : int, optional
            Defaults to 25
        sensor_key_list : list of str, optional
            list containing the names of the sensors that should feed data to the
            Redundancy Agent. Defaults to None
        n_pr : int, optional
            size of the batch of data that is handled at a time by the Redundancy Agent.
            Defaults to 1
        problim : float, optional
            limit probability used for consistency evaluation. Defaults to .9
        calc_type : str, optional
            calculation type: 'lcs' or 'lcss'. Defaults to 'lcs'
        """

        if sensor_key_list is None:
            sensor_key_list = []
        super().init_parameters(input_data_maxlen=25, output_data_maxlen=25)
        self.metadata = MetaData(
            device_id="RedAgent01",
            time_name="time",
            time_unit="s",
            quantity_names="m",
            quantity_units="kg",
            misc="nothing",
        )

        self.calc_type = calc_type
        self.sensor_key_list = sensor_key_list
        self.n_pr = n_pr
        self.problim = problim

        self.set_output_data(channel="default", metadata=self.metadata)

    def init_lcss_parameters(self, fsam, f1, f2, ampl_ratio, phi1, phi2):
        """Additional parameters used for this particular example

        Provides the prior knowledge needed to make the information contained in the
        data redundant. This method sets up the vector **a** and matrix *A* for the
        system **y** = **a** + *A* * **x**.

        Parameters
        ----------
        fsam : float
            sampling frequency
        f1 : float
            first frequency of interest in signal
        f2 : float
            second frequency of interest in signal
        ampl_ratio : float
            ratio of the amplitudes of the two frequency components
        phi1 : float
            initial phase of first frequency component
        phi2 : float
            initial phase of second frequency component
        """
        # set-up vector a_arr and matrix a_arr2d for redundancy method
        id_mat = np.identity(self.n_pr)
        id_fft = np.fft.fft(id_mat)

        n_pr_floor = self.n_pr // 2
        bfft = id_fft[:n_pr_floor, :].real / n_pr_floor
        cfft = -id_fft[:n_pr_floor, :].imag / n_pr_floor

        c1 = 1 / np.cos(phi1)
        c2 = 1 / np.sin(-phi1)
        c3 = ampl_ratio / np.cos(phi2)
        c4 = ampl_ratio / np.sin(-phi2)

        ind_freq1 = f1 * self.n_pr // fsam
        ind_freq2 = f2 * self.n_pr // fsam

        a_row1 = c1 * bfft[ind_freq1, :].reshape((1, -1))
        a_row2 = c2 * cfft[ind_freq1, :].reshape((1, -1))
        a_row3 = c3 * bfft[ind_freq2, :].reshape((1, -1))
        a_row4 = c4 * cfft[ind_freq2, :].reshape((1, -1))

        self.a_arr2d = np.concatenate((a_row1, a_row2, a_row3, a_row4), axis=0)
        self.a_arr = np.zeros(shape=(4, 1))

    def agent_loop(self):
        """Model the agent's behaviour

        On state *Running* the agent will extract sample by sample the input data
        streams content and push it via invoking
        :py:func:`send_output <agentMET4FOF.agents.AgentMET4FOF.send_output>`.
        """
        if self.current_state == "Running":
            key_list = [
                key for key in self.sensor_key_list if key in self.buffer.keys()
            ]
            n_sensors = len(key_list)
            if n_sensors != len(self.sensor_key_list):  # expected number of sensors
                print(
                    "Not all sensors were present in the buffer."
                    "Not evaluating the data."
                )
                return 0

            for key in key_list:
                if self.buffer[key].shape[0] < self.n_pr:
                    print(
                        "Buffer size is ",
                        self.buffer[key].shape[0],
                        ", which is less than ",
                        self.n_pr,
                        ".",
                    )
                    print("Not enough data for redundancy agent evaluation.")
                    return 0

            buff = self.buffer.popleft(
                self.n_pr
            )  # take n_pr entries out from the buffer

            t_data_arr2d = np.full(shape=(self.n_pr, n_sensors), fill_value=np.nan)
            ut_data_arr2d = np.full(shape=(self.n_pr, n_sensors), fill_value=np.nan)
            x_data_arr2d = np.full(shape=(self.n_pr, n_sensors), fill_value=np.nan)
            ux_data_arr2d = np.full(shape=(self.n_pr, n_sensors), fill_value=np.nan)
            i_sensor = 0

            for key in key_list:
                data_arr = buff[key]
                t_data_arr2d[:, i_sensor] = data_arr[:, 0]
                ut_data_arr2d[:, i_sensor] = data_arr[:, 1]
                x_data_arr2d[:, i_sensor] = data_arr[:, 2]
                ux_data_arr2d[:, i_sensor] = data_arr[:, 3]
                i_sensor = i_sensor + 1

            data = np.full(shape=(self.n_pr, 4), fill_value=np.nan)
            if self.calc_type == "lcs":
                for i_pnt in range(self.n_pr):
                    y_arr = np.array(x_data_arr2d[i_pnt, :])
                    y_arr = y_arr.reshape((n_sensors, 1))
                    vy_arr2d = np.zeros(shape=(n_sensors, n_sensors))
                    for i_sensor in range(n_sensors):
                        vy_arr2d[i_sensor, i_sensor] = np.square(
                            ux_data_arr2d[i_pnt, i_sensor]
                        )

                    n_solutions, ybest, uybest, chi2obs, indkeep = self.calc_lcs(
                        y_arr, vy_arr2d, self.problim
                    )
                    if n_solutions == 1:
                        if isinstance(ybest, np.ndarray):
                            ybest = ybest[0]
                        data[i_pnt, :] = np.array(
                            [
                                t_data_arr2d[i_pnt, 0],
                                ut_data_arr2d[i_pnt, 0],
                                ybest,
                                uybest,
                            ]
                        )
                    else:  # only return the first solution
                        data[i_pnt, :] = np.array(
                            [
                                t_data_arr2d[i_pnt, 0],
                                ut_data_arr2d[i_pnt, 0],
                                ybest[0],
                                uybest[0],
                            ]
                        )
            elif self.calc_type == "lcss":
                # lcss applied to one data vector (required input)
                # Sum the signals to get one signal
                x_data_arr = np.sum(x_data_arr2d, axis=1)
                x_data_arr = x_data_arr.reshape((len(x_data_arr), 1))
                ux2_data_arr = np.sum(np.square(ux_data_arr2d), axis=1)
                vx_arr2d = np.zeros((self.n_pr, self.n_pr))
                for i_val in range(self.n_pr):
                    vx_arr2d[i_val, i_val] = ux2_data_arr[i_val]

                n_solutions, ybest, uybest, chi2obs, indkeep = self.calc_lcss(
                    self.a_arr, self.a_arr2d, x_data_arr, vx_arr2d, self.problim
                )
                print("calc lcss finished")
                print("n_solutions: ", n_solutions)
                print("ybest: ", ybest)
                print("uybest: ", uybest)
                if n_solutions == 1:  # time stamp is latest value
                    if isinstance(ybest, np.ndarray):
                        ybest = ybest[0]
                    data = np.array(
                        [t_data_arr2d[-1, 0], ut_data_arr2d[-1, 0], ybest, uybest]
                    )
                else:  # only return the first solution
                    data = np.array(
                        [t_data_arr2d[-1, 0], ut_data_arr2d[-1, 0], ybest[0], uybest[0]]
                    )

            # Send the data
            if len(data.shape) == 1:
                data = data.reshape((1, len(data)))

            self.set_output_data(channel="default", data=data)
            super().agent_loop()

    @staticmethod
    def calc_consistent_estimates_no_corr(y_arr2d, uy_arr2d, prob_lim):
        """Calculation of consistent estimate for sets of estimates y_ij

        The y_ij (contained in y_arr2d) are the elements of Y, where each
        set contains n_estims estimates. The uncertainties are assumed to be
        independent and given in uy_arr2d. The consistency test is using limit
        probability limit prob_lim. For each set of estimates, the best estimate,
        uncertainty, observed chi-2 value and a flag if the
        provided estimates were consistent given the model are given as output.

        Parameters
        ----------
        y_arr2d : np.ndarray of size (n_rows, n_estimates)
            each row contains m=n_estimates independent estimates of a measurand
        uy_arr2d : np.ndarray of size (n_rows, n_estimates)
            each row contains the standard uncertainty u(y_ij) of y_ij = y_arr2d[i,j]
        prob_lim : float
            limit probability used in consistency test. Typically 0.95.

        Returns
        -------
        isconsist_arr : bool array of shape (n_rows)
            indicates for each row if the n_estimates are consistent or not
        ybest_arr : float or np.ndarray of float in shape (n_rows)
            contains the best estimate for each row of individual estimates
        uybest_arr : float or np.ndarray of float in shape (n_rows)
            contains the uncertainty associated with each best estimate for each row
            of *y_arr2d*
        chi2obs_arr : np.ndarray of float in shape (n_rows)
            observed chi-squared value for each row
        """

        if len(y_arr2d.shape) > 1:
            n_sets = y_arr2d.shape[0]
        else:
            n_sets = 1

        n_estims = y_arr2d.shape[-1]  # last dimension is number of estimates
        chi2_lim = chi2.ppf(1 - prob_lim, n_estims - 1)
        uy2inv_arr2d = 1 / np.power(uy_arr2d, 2)
        uy2best_arr = 1 / np.sum(uy2inv_arr2d, -1)
        uybest_arr = np.sqrt(uy2best_arr)
        ybest_arr = np.sum(y_arr2d * uy2inv_arr2d, -1) * uy2best_arr

        if n_sets > 1:
            ybest_arr = ybest_arr.reshape(
                n_sets, 1
            )  # make a column vector of ybest_arr

        chi2obs_arr = np.sum(
            np.power(
                (y_arr2d - np.broadcast_to(ybest_arr, (n_sets, n_estims))) / uy_arr2d, 2
            ),
            -1,
        )
        isconsist_arr = chi2obs_arr <= chi2_lim

        return isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr

    @staticmethod
    def print_output_single(isconsist, ybest, uybest, chi2obs):
        """Print the output of a single row of the calculate_best_estimate function

        Parameters
        ----------
        isconsist : bool
            Indicates if provided estimates were consistent
        ybest : float
            best estimate
        uybest : float
            uncertainty of best estimate
        chi2obs : float
            observed value of chi-squared
        """
        print("\tThe observed chi-2 value is %3.3f." % chi2obs)

        if isconsist:
            print("\tThe provided estimates (input) were consistent.")
        else:
            print("\tThe provided estimates (input) were not consistent.")

        print(f"\tThe best estimate is {ybest:3.3f} with uncertainty {uybest:3.3f}.\n")

    def print_output_cbe(self, isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr):
        """Function to print the full output of calc_best_estimate.

        Parameters
        ----------
        isconsist_arr : bool array of shape (n_rows)
            indicates for each row if the n_estimates are consistent or not
        ybest_arr : np.ndarray of floats in shape (n_rows)
            contains the best estimate for each row of individual estimates
        uybest_arr : np.ndarray of floats in shape (n_rows)
            contains the uncertainty associated with each best estimate for each row
            of *y_arr2d*
        chi2obs_arr : np.ndarray of floats in shape (n_rows)
            observed chi-squared value for each row
        """
        if len(ybest_arr.shape) == 0:
            self.print_output_single(isconsist_arr, ybest_arr, uybest_arr, chi2obs_arr)
        else:
            n_sets = ybest_arr.shape[0]
            print(f"There are {n_sets:.0f} sets with estimates of the measurand.")
            for i_set in range(n_sets):
                print(f"The result of set {i_set:.0f} is:")
                self.print_output_single(
                    isconsist_arr.item(i_set),
                    ybest_arr.item(i_set),
                    uybest_arr.item(i_set),
                    chi2obs_arr.item(i_set),
                )

    @staticmethod
    def calc_best_estimate(y_arr, vy_arr2d, problim):
        """Calculate the best estimate for a set of estimates with uncertainties

        Additionally determine if the set of estimates are consistent using a provided
        limit probability.

        Parameters
        ----------
        y_arr : np.ndarray of shape (n)
            vector of estimates of a measurand Y
        vy_arr2d : np.ndarray of shape (n, n)
            uncertainty matrix associated with y_arr
        problim : float
            probability limit used for assessing the consistency of the estimates.
            Typically, problim equals 0.95.

        Returns
        -------
        isconsist : bool
            indicator whether provided estimates are consistent in view of *problim*
        ybest : float
            best estimate of measurand
        uybest : float
            uncertainty associated with *ybest*
        chi2obs : float
            observed value of chi-squared, used for consistency evaluation
        """

        print(f"cbe y_arr = {y_arr}")
        n_estims = len(y_arr)

        if n_estims == 1:
            isconsist = True
            ybest = y_arr[0]
            uybest = np.sqrt(vy_arr2d[0, 0])
            chi2obs = 0.0
        else:
            e_arr = np.ones(n_estims)
            vyinve_arr = np.linalg.solve(vy_arr2d, e_arr)
            uy2 = 1 / np.dot(e_arr, vyinve_arr)
            uybest = np.sqrt(uy2)
            ybest = np.dot(vyinve_arr, y_arr) * uy2
            yred_arr = y_arr - ybest
            chi2obs = np.dot(
                yred_arr.transpose(), np.linalg.solve(vy_arr2d, yred_arr)
            )  # check need for transpose
            chi2lim = chi2.ppf(problim, n_estims - 1)
            isconsist = chi2obs <= chi2lim

        return isconsist, ybest, uybest, chi2obs

    @staticmethod
    def get_combination(values, n_keep, certain_combinations_index):
        """Return a certain subset of n_keep elements in a given array

        Parameters
        ----------
        values : np.ndarray
            original values
        n_keep : int
            number of elements in subset
        certain_combinations_index : int
            the index of the desired combination as a result of a call
            of ``combinations(values, n_keep)``
        """
        subsets = combinations(values, n_keep)
        i_subset = -1

        for subset in subsets:
            i_subset += 1
            if i_subset == certain_combinations_index:
                return np.array(list(subset))

    def calc_lcs(self, y_arr, vy_arr2d, problim):
        """Calculate the best estimate of a measurand with associated uncertainty matrix

        Parameters
        ----------
        y_arr: np.ndarray of shape (n)
            vector with estimates of the measurand
        vy_arr2d: np.ndarray of shape (n, n)
            uncertainty matrix of the vector y_arr
        problim: float
            limit probability used in the consistency evaluation. Typically 0.95.

        Returns
        -------
        n_solutions : int or np.ndarray of ints
            number of solutions
        ybest : float or np.ndarray of floats
            best estimate
        uybest : float or np.ndarray of floats
            standard uncertainty of best estimate
        chi2obs : float or np.ndarray of floats
            observed chi-squared value
        indkeep : np.ndarray of shape (n) or (n_sols, n)
            indices of kept estimates
        """
        n_solutions = 0
        indkeep = np.nan

        isconsist, ybest, uybest, chi2obs = self.calc_best_estimate(
            y_arr, vy_arr2d, problim
        )
        n_estims = len(y_arr)
        estim_arr = np.arange(n_estims)
        n_remove = 0

        if isconsist:  # set the other return variables
            n_solutions = 1
            indkeep = estim_arr

        while not isconsist:
            n_remove += 1
            subsets = combinations(estim_arr, n_estims - n_remove)
            n_subsets = comb(n_estims, n_remove, exact=True)
            isconsist_arr = np.full(n_subsets, np.nan)
            ybest_arr = np.full(n_subsets, np.nan)
            uybest_arr = np.full(n_subsets, np.nan)
            chi2obs_arr = np.full(n_subsets, np.nan)
            i_subset = -1

            for subset in subsets:
                i_subset += 1
                sublist = list(subset)
                yred_arr = y_arr[sublist]
                vyred_arr2d = vy_arr2d[np.ix_(sublist, sublist)]
                (
                    isconsist_arr[i_subset],
                    ybest_arr[i_subset],
                    uybest_arr[i_subset],
                    chi2obs_arr[i_subset],
                ) = self.calc_best_estimate(yred_arr, vyred_arr2d, problim)

            # Find smallest chi2obs value amongst all subsets. If multiple possibilities
            # exist, return them all
            indmin = np.argmin(chi2obs_arr)

            if isconsist_arr[indmin]:
                # consistent solution found (otherwise isconsist remains false and the
                # while loop continues)
                isconsist = True
                chi2obs = chi2obs_arr[indmin]  # minimum chi2obs value
                indmin = np.where(chi2obs_arr == chi2obs)[
                    0
                ]  # list with all indices with minimum chi2obs value
                n_solutions = len(indmin)

                if n_solutions == 1:
                    ybest = ybest_arr[indmin[0]]
                    uybest = uybest_arr[indmin[0]]
                    indkeep = self.get_combination(
                        estim_arr, n_estims - n_remove, indmin
                    )
                else:  # multiple solutions exist, the return types become arrays
                    ybest = np.full(n_solutions, np.nan)
                    uybest = np.full(n_solutions, np.nan)
                    indkeep = np.full((n_solutions, n_estims - n_remove), np.nan)

                    for i_sol in range(n_solutions):
                        ybest[i_sol] = ybest_arr[indmin[i_sol]]
                        uybest[i_sol] = uybest_arr[indmin[i_sol]]
                        indkeep[i_sol] = self.get_combination(
                            estim_arr, n_estims - n_remove, indmin[i_sol]
                        )

        return n_solutions, ybest, uybest, chi2obs, indkeep

    def on_received_message(self, message):
        """Handle incoming data from 'default' channels

        Store 'default' data into an internal buffer.

        Parameters
        ----------
        message : dict
             Only acceptable channel value is 'default'.
        """
        if message["channel"] == "default":
            self.buffer_store(agent_from=message["from"], data=message["data"])
        return 0

    @staticmethod
    def print_output_lcs(n_solutions, ybest, uybest, chi2obs, indkeep, y_arr):
        """Method to print the output of the method :func:`calc_lcs`

        Parameters
        ----------
        n_solutions : int
            number of best solutions
        ybest : float or np.ndarray of shape (n_sols)
            best estimate or vector of best estimates
        uybest : float or np.ndarray of shape (n_sols)
            standard uncertainty of best estimate or vector with standard uncertainty
            of best estimates
        chi2obs : float
            observed chi-squared value of all best solutions
        indkeep : np.ndarray of shape (n) or (n_sols, n)
            indices of retained estimates of y_arr for the calculation of the best
            estimate ybest
        y_arr : np.ndarray of shape (n)
            individual estimates of measurand
        """
        n_estims = len(y_arr)
        n_keep = indkeep.shape[
            -1
        ]  # number of retained estimates in the best solution(s)

        if n_solutions == 1:
            print(
                f"calc_lcs found a unique solution with chi2obs = {chi2obs:4.4f} using"
                f"{n_keep:.0f} of the provided {n_estims:.0f} estimates."
            )
            print(f"\ty = {ybest:4.4f}, u(y) = {uybest:4.4f}")
            print(f"\tIndices and values of retained provided estimates:", end=" ")

            for ind in indkeep[:-1]:
                indint = int(ind)
                print(f"y[{indint:.0f}]= {y_arr[indint]:2.2f}", end=", ")

            indint = int(indkeep[-1])
            print(f"y[{indint:.0f}]= {y_arr[indint]:2.2f}.\n")
        else:
            print(
                f"calc_lcs found {n_solutions:.0f} equally good solutions with chi2obs "
                f"= {chi2obs:4.4f} using {n_keep:.0f} of the provided {n_estims:.0f} "
                f"estimates."
            )

            for i_sol in range(n_solutions):
                print(f"\tSolution {i_sol:.0f} is:")
                print(f"\ty = {ybest[i_sol]:4.4f}, u(y) = {uybest[i_sol]:4.4f}")
                print("\tIndices and values of retained provided estimates:", end=" ")

                for ind in indkeep[i_sol][:-1]:
                    indint = int(ind)
                    print("y[%d]= %2.2f" % (indint, y_arr[indint]), end=", ")

                indint = int(indkeep[i_sol][-1])
                print("y[%d]= %2.2f.\n" % (indint, y_arr[indint]))

        return

    @staticmethod
    def ind_reduce_a(a_arr2d, epszero):
        """Returns the index of a linear dependent row of a matrix A

        The motivation for this is, that this row does not contribute any new
        information to the system.

        Parameters
        ----------
        a_arr2d : np.ndarray
            The matrix to be reduced as 2-dimensional array
        epszero : float
            some small constant used for checking equality to zero

        Returns
        -------
        int
            the index of the last row that can be taken out
        """
        if a_arr2d.shape[0] <= np.linalg.matrix_rank(a_arr2d):
            raise SystemMatrixNotReducibleError("A cannot be reduced!")
        # Find a solution of A' * b = 0.
        u, s, vh = np.linalg.svd(np.transpose(a_arr2d))
        b = vh[-1, :]
        indrem = np.where(abs(b) > epszero)[
            0
        ]  # Remove a row corresponding to a non-zero entry in b.

        if len(indrem) == 0:
            raise ValueError("b is a zero vector!")

        indrem = indrem[-1]
        return indrem

    @staticmethod
    def reduce_vx(x_arr, vx_arr2d, a_arr, a_arr2d, epszero):
        """Reduce the system if matrix Vx is not of full rank

        This might be ambiguous, as constant sensor values or offsets have to be
        estimated and are not known.

        Parameters
        ----------
        x_arr : np.ndarray
            The vector x to be reduced
        vx_arr2d : np.ndarray
            The matrix Vx to be reduced as 2-dimensional array
        a_arr : np.ndarray
            The vector a to be reduced
        a_arr2d : np.ndarray
            The matrix A to be reduced as 2-dimensional array
        epszero : float
            some small constant used for checking equality to zero

        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            The reduced matrices and vectors xred_arr, vxred_arr2d, ared_arr, ared_arr2d
        """
        if vx_arr2d.shape[0] <= np.linalg.matrix_rank(vx_arr2d, epszero):
            print("Vx cannot be reduced any further!")
            return
        # Remove one sensor from Vx, A and x that is a linear combination of the other
        # sensors. Find a solution of Vx * b = 0.
        u, s, vh = np.linalg.svd(vx_arr2d)
        b = vh[-1, :]  # bottom row of vh is orthogonal to Vx

        if abs(np.dot(b, x_arr)) > epszero:
            raise SensorsNotLinearlyIndependentError(
                "Sensors in x should be linearly independent with b^T * x = 0, but"
                "this is not the case!"
            )

        indrem = np.where(abs(b) > epszero)[
            0
        ]  # Remove a sensor corresponding to a non-zero entry in b.
        if len(indrem) == 0:
            raise ValueError("b is the zero vector!")

        indrem = indrem[-1]  # take out the last sensor
        indsenskeep = np.concatenate(
            (np.arange(indrem), np.arange(indrem, vx_arr2d.shape[0]))
        )
        vxred_arr2d = vx_arr2d[indsenskeep, indsenskeep]
        xred_arr = x_arr[indsenskeep]
        # Update A by removing the sensor and updating the system of equations

        ared_arr2d = a_arr2d - a_arr2d[:, indrem] / b[indrem] * np.transpose(b)
        if max(abs(ared_arr2d[:, indrem])) > epszero:
            print(ared_arr2d)
            raise ColumnNotZeroError(f"Column {indrem} should be zero by now!")

        ared_arr2d = a_arr2d[:, indsenskeep]  # remove the zero column from A
        ared_arr = a_arr + np.dot(
            a_arr2d - ared_arr2d, x_arr
        )  # adapt vector a_arr such that the vector of estimates y = a + A*x remains
        # the same

        return xred_arr, vxred_arr2d, ared_arr, ared_arr2d

    def calc_best_est_lin_sys(self, a_arr, a_arr2d, x_arr, vx_arr2d, problim):
        """Calculate the best estimate of a linear system **y** = **a** + A * **x**

        Additionally determine if the inputs are consistent in view of *problim*.

        Parameters
        ----------
        a_arr : np.ndarray of shape (n_estimates)
            vector **a** of linear system **y** = **a** + A * **x**
        a_arr2d : np.ndarray of shape (n_estimates, n_sensors)
            matrix A of linear system **y** = **a** + A * **x**
        x_arr : np.ndarray of shape (n_sensors)
            vector with sensor values, vector **x** of linear system **y** = **a** +
            A * **x**
        vx_arr2d : np.ndarray of shape (n_sensors, n_sensors)
            uncertainty matrix associated with vector x_arr
        problim : float
            probability limit used for consistency evaluation. Typically 0.95.

        Returns
        -------
        isconsist : bool
            indicator whether provided estimates are consistent in view of *problim*
        ybest : float
            best estimate
        uybest : float
            standard uncertainty of best estimate
        chi2obs : float
            observed chi-squared value
        """
        print("start calc_best_est_lin_sys")
        epszero = 1e-10  # some small constant used for some checks

        # The main procedure only works when vy_arr2d has full rank. Therefore first
        # a_arr, a_arr2d and vx_arr2d need to be reduced such that vy_arr2d will have
        # full rank.
        xred_arr = x_arr
        vxred_arr2d = vx_arr2d
        ared_arr = a_arr
        ared_arr2d = a_arr2d

        # Reduce the system if the covariance matrix vx_arr2d is rank deficient.
        while np.linalg.matrix_rank(vxred_arr2d) < vxred_arr2d.shape[0]:
            print(
                "Reducing Vx. No of rows = ",
                vxred_arr2d.shape[0],
                ", rank = ",
                np.linalg.matrix_rank(vxred_arr2d),
            )
            [xred_arr, vxred_arr2d, ared_arr, ared_arr2d] = self.reduce_vx(
                xred_arr, vxred_arr2d, ared_arr, ared_arr2d, epszero
            )

        # Reduce the system if a_arr2d has more rows than its rank.
        while ared_arr2d.shape[0] > np.linalg.matrix_rank(ared_arr2d):
            print(
                "Reducing A. No of rows = ",
                ared_arr2d.shape[0],
                ", rank = ",
                np.linalg.matrix_rank(ared_arr2d),
            )
            print(f"ared_arr2d: {ared_arr2d}")
            ind_rem = self.ind_reduce_a(ared_arr2d, epszero)
            n_rows = ared_arr2d.shape[0]
            indrowskeep = np.concatenate(
                (np.arange(0, ind_rem), np.arange(ind_rem + 1, n_rows))
            )
            ared_arr = ared_arr[indrowskeep]
            ared_arr2d = ared_arr2d[
                indrowskeep,
            ]

        # calculate y vector and Vy matrix
        print("ared_arr2d: ", ared_arr2d)
        print("ared_arr: ", ared_arr.shape)
        print("ared_arr2d: ", ared_arr2d.shape)
        print("xred_arr: ", xred_arr.shape)
        y_arr = ared_arr + np.dot(ared_arr2d, xred_arr)
        vy_arr2d = np.matmul(
            np.matmul(ared_arr2d, vxred_arr2d), np.transpose(ared_arr2d)
        )

        # try to calculate a consistent solution with these y and Vy.
        isconsist, ybest, uybest, chi2obs = self.calc_best_estimate(
            y_arr, vy_arr2d, problim
        )
        return isconsist, ybest, uybest, chi2obs

    def calc_lcss(self, a_arr, a_arr2d, x_arr, vx_arr2d, problim):
        """Calculation of the largest consistent subset of sensor values

        Additionally the implied best estimate is returned.

        Parameters
        ----------
        a_arr : np.ndarray of shape (n_estimates)
            vector **a** of linear system **y** = **a** + A * **x**
        a_arr2d : np.ndarray of shape (n_estimates, n_sensors)
            matrix A of linear system **y** = **a** + A * **x**
        x_arr : np.ndarray of shape (n_sensors)
            vector with sensor values
            vector **x** of linear system **y** = **a** + A * **x**
        vx_arr2d : np.ndarray of shape (n_sensors, n_sensors)
            uncertainty matrix associated with vector x_arr
        problim : float
            probability limit used for consistency evaluation. Typically 0.95

        Returns
        -------
        n_solutions: int or np.ndarray of ints
            number of solutions
        isconsist: bool or np.ndarray of bool
            indicator whether provided estimates are consistent in view of *problim*
        ybest: float or np.ndarray of floats
            best estimate
        uybest: float or np.ndarray of floats
            standard uncertainty of best estimate
        chi2obs: float or np.ndarray of floats
            observed chi-squared value
        """
        n_solutions = 0
        indkeep = np.nan

        print("start calc_lcss")
        epszero = 1e-7  # epsilon for rank check
        eps_chi2 = 1e-7  # epsilon for chi2 equivalence check

        isconsist, ybest, uybest, chi2obs = self.calc_best_est_lin_sys(
            a_arr, a_arr2d, x_arr, vx_arr2d, problim
        )
        n_sens = len(x_arr)
        sens_arr = np.arange(n_sens)
        n_remove = 0

        if isconsist:  # set the other return variables
            n_solutions = 1
            indkeep = sens_arr

        # no consistent solution, remove sensors 1 by 1, 2 by 2, etc.
        while not isconsist:
            n_remove += 1
            subsets = combinations(sens_arr, n_sens - n_remove)
            n_subsets = comb(n_sens, n_remove, exact=True)
            isconsist_arr = np.full(n_subsets, np.nan)
            ybest_arr = np.full(n_subsets, np.nan)
            uybest_arr = np.full(n_subsets, np.nan)
            chi2obs_arr = np.full(n_subsets, np.nan)
            i_subset = -1
            for subset in subsets:
                i_subset += 1
                sublistsenskeep = list(subset)
                xred_arr = x_arr[sublistsenskeep]
                vxred_arr2d = vx_arr2d[:, sublistsenskeep]
                vxred_arr2d = vxred_arr2d[sublistsenskeep, :]
                boolremove_arr = np.full(n_sens, True)
                boolremove_arr[sublistsenskeep] = False
                if (
                    np.linalg.matrix_rank(
                        np.concatenate(
                            (
                                a_arr2d[:, boolremove_arr],
                                np.ones((a_arr2d.shape[0], 1)),
                            ),
                            axis=1,
                        ),
                        epszero,
                    )
                    == np.linalg.matrix_rank(a_arr2d[:, boolremove_arr], epszero)
                ):
                    # there is no vector c such that c' * ones = 1 and c' * ai = 0 at
                    # the same time. Thus this combination of sensors cannot be
                    # removed as a group from the matrix A.
                    isconsist_arr[i_subset] = False
                    continue  # continue with next subset

                ared_arr2d = np.concatenate(
                    (a_arr2d[:, boolremove_arr], a_arr2d[:, sublistsenskeep]), axis=1
                )  # move the columns corresponding to sensors to be taken out to the
                # front
                q, r = np.linalg.qr(ared_arr2d)
                q1 = q[
                    :, n_remove:
                ]  # these (n_sens-n_remove) columns are orthogonal to the first n
                # remove columns of ared_arr2d
                s = np.sum(
                    q1, axis=0
                )  # column sums might be zero which is a problem for normalization
                # to unit sum
                indzero = np.where(np.abs(s) < epszero)[0]
                if len(indzero) > 0:
                    indnonzero = np.full(n_sens - n_remove, True)  # all True array
                    indnonzero[indzero] = False  # positions that are zero are false
                    indnonzero = np.where(
                        indnonzero
                    )  # conversion to indices instead of boolean array
                    if len(indnonzero) == 0:
                        print("ERROR: All columns have zero sum!")
                    b = q1[
                        :, indnonzero[0]
                    ]  # b is column vector with no zero column sum
                    for i_zero in range(len(indzero)):
                        q1[:, indzero[i_zero]] = (
                            q1[:, indzero[i_zero]] + b
                        )  # add b to prevent zero column sum
                q1 = q1 / np.sum(
                    q1, axis=0
                )  # unit column sums, in order not to introduce a bias in the estimate
                # of the measurand

                ared_arr2d = np.matmul(np.transpose(q1), ared_arr2d)
                ared_arr = np.matmul(np.transpose(q1), a_arr)
                # The columns of matrix ared_arr2d are still in the wrong order compared
                # to the order of the sensors
                ared2_arr2d = np.full_like(ared_arr2d, np.nan)
                ared2_arr2d[:, boolremove_arr] = ared_arr2d[
                    :, :n_remove
                ]  # columns 0, 1, ..., (n_remove-1)
                ared2_arr2d[:, np.invert(boolremove_arr)] = ared_arr2d[
                    :, n_remove:
                ]  # columns n_remove, ..., (n_sens-1)
                ared_arr2d = ared2_arr2d

                if np.linalg.norm(ared_arr2d[:, boolremove_arr]) > epszero:
                    raise ColumnNotZeroError(
                        f"These columns of A should be zero by now!"
                    )

                ared_arr2d = ared_arr2d[
                    :, sublistsenskeep
                ]  # reduce the matrix A by removing the appropriate columns of A,
                # which are zero anyway.

                (
                    isconsist_arr[i_subset],
                    ybest_arr[i_subset],
                    uybest_arr[i_subset],
                    chi2obs_arr[i_subset],
                ) = self.calc_best_est_lin_sys(
                    ared_arr, ared_arr2d, xred_arr, vxred_arr2d, problim
                )

            # After analyzing all subset, find the smallest chi2obs value amongst all
            # subsets.
            # If multiple possibilities exist, return them all
            indmin = np.argmin(chi2obs_arr)
            if isconsist_arr[indmin]:
                # consistent solution found (otherwise isconsist remains false and
                # the while loop continues)
                isconsist = True
                chi2obs = chi2obs_arr[indmin]  # minimum chi2obs value
                indmin = np.where(np.abs(chi2obs_arr - chi2obs) < eps_chi2)[
                    0
                ]  # list with all indices with minimum chi2obs value
                n_solutions = len(indmin)
                if n_solutions == 1:
                    ybest = ybest_arr[indmin[0]]
                    uybest = uybest_arr[indmin[0]]
                    indkeep = self.get_combination(
                        sens_arr, n_sens - n_remove, indmin
                    )  # indices of kept estimates
                else:  # multiple solutions exist, the return types become arrays
                    ybest = np.full(n_solutions, np.nan)
                    uybest = np.full(n_solutions, np.nan)
                    indkeep = np.full((n_solutions, n_sens - n_remove), np.nan)
                    for i_sol in range(n_solutions):
                        ybest[i_sol] = ybest_arr[indmin[i_sol]]
                        uybest[i_sol] = uybest_arr[indmin[i_sol]]
                        indkeep[i_sol] = self.get_combination(
                            sens_arr, n_sens - n_remove, indmin[i_sol]
                        )
        return n_solutions, ybest, uybest, chi2obs, indkeep

    @staticmethod
    def print_input_lcss(x_arr, vx_arr2d, a_arr, a_arr2d, problim):
        """Prints the input parameters of the method

        Parameters
        ----------
        x_arr : np.ndarray of shape (n_sensors)
            vector with sensor values, vector **x** of linear system **y** = **a** +
            A * **x**
        vx_arr2d : np.ndarray of shape (n_sensors, n_sensors)
            uncertainty matrix associated with vector x_arr
        a_arr : np.ndarray of shape (n_estimates)
            vector **a** of linear system **y** = **a** + A * **x**
        a_arr2d : np.ndarray of shape (n_estimates, n_sensors)
            matrix A of linear system **y** = **a** + A * **x**
        problim : float
            probability limit used for consistency evaluation. Typically 0.95
        """
        print(
            f"""INPUT of lcss function call:
        Vector a of linear system: a_arr = {a_arr}
        Matrix A of linear system: a_arr2d = {a_arr2d}
        Vector x with sensor values: x_arr = {x_arr}
        Covariance matrix Vx of sensor values: vx_arr2d = {vx_arr2d}
        Limit probability for chi-squared test: p = {problim}"""
        )

    @staticmethod
    def print_output_lcss(n_solutions, ybest, uybest, chi2obs, indkeep, x_arr, a_arr2d):
        """Prints the outputs of the method :meth:`calc_lcss`

        Parameters
        ----------
        n_solutions : int or np.ndarray of ints
            number of solutions
        ybest : float or np.ndarray of floats
            best estimate
        uybest : float or np.ndarray of floats
            standard uncertainty of best estimate
        chi2obs : float or np.ndarray of floats
            observed chi-squared value
        indkeep : np.ndarray of int
            indices of kept estimates
        x_arr : np.ndarray of shape (n_estimates)
            vector **a** of linear system **y** = **a** + A * **x**
        a_arr2d : np.ndarray of shape (n_estimates, n_sensors)
            matrix A of linear system **y** = **a** + A * **x**

        Returns
        -------
        None
        """
        n_sensors = len(x_arr)
        n_eq = a_arr2d.shape[0]
        n_keep = indkeep.shape[
            -1
        ]  # number of retained estimates in the best solution(s)
        print(
            f"Provided number of sensors (or sensor values) was {n_sensors} and "
            f"number of equations was {n_eq}."
        )
        if n_solutions == 1:
            print(
                "calc_lcss found a unique solution with chi2obs = %4.4f using %d of "
                "the provided %d sensor values." % (chi2obs, n_keep, n_sensors)
            )
            print("\ty = %4.4f, u(y) = %4.4f" % (ybest, uybest))
            print("\tIndices and values of retained provided sensor values:", end=" ")
            for ind in indkeep[:-1]:
                indint = int(ind)
                print("x[%d]= %2.2f" % (indint, x_arr[indint]), end=", ")
            indint = int(indkeep[-1])
            print("x[%d]= %2.2f.\n" % (indint, x_arr[indint]))
        else:
            print(
                "calc_lcss found %d equally good solutions with chi2obs = %4.4f "
                "using %d of the provided %d sensor values."
                % (n_solutions, chi2obs, n_keep, n_eq)
            )
            for i_sol in range(n_solutions):
                print("\tSolution %d is:" % i_sol)
                print("\ty = %4.4f, u(y) = %4.4f" % (ybest[i_sol], uybest[i_sol]))
                print(
                    "\tIndices and values of retained provided sensor values:", end=" "
                )
                for ind in indkeep[i_sol][:-1]:
                    indint = int(ind)
                    print("x[%d]= %2.2f" % (indint, x_arr[indint]), end=", ")
                indint = int(indkeep[i_sol][-1])
                print("x[%d]= %2.2f.\n" % (indint, x_arr[indint]))
        return
