from agentMET4FOF.agents import AgentMET4FOF
import math
from scipy.spatial.ckdtree import cKDTree
import numpy as np


class RendezvousAgent(AgentMET4FOF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rendezvous_steps_list_buffer = None
        self.spatial_threshold = None
        self.time_threshold = None

    def init_parameters(self):
        pass

    def get_time_distance(self, t1, t2):
        return math.fabs(t2 - t1)

    def get_spatial_distance(self, coordinates_1, coordinates_2):
        return math.sqrt(
            (coordinates_2[0] - coordinates_1[0]) ** 2
            + (coordinates_2[1] - coordinates_1[1]) ** 2
        )

    def find_neighbors(
            self,
            sensor_one_positions,
            sensor_one_times,
            sensor_two_positions,
            sensor_two_times,
            spatial_threshold,
            time_threshold
    ):

        positions_neighbor_pairs_set = self._find_isotropic_neighbors(
            sensor_one_positions, sensor_two_positions, spatial_threshold
        )
        times_neighbor_pairs_set = self._find_isotropic_neighbors(
            sensor_one_times, sensor_two_times, time_threshold
        )

        neighbor_pairs_set = positions_neighbor_pairs_set.intersection(times_neighbor_pairs_set)
        return neighbor_pairs_set

    def find_all_neighbors(self, sensor_positions_list, sensor_times_list, spatial_threshold, time_threshold):
        assert len(sensor_positions_list) == len(sensor_times_list), (len(sensor_positions_list), len(sensor_times_list))
        n_sensors = len(sensor_positions_list)

        neighbor_pairs_set = set()

        for i in range(n_sensors):
            for j in range(n_sensors):
                if i <= j:
                    continue

                _neighbor_pairs_set_two_sensors = self.find_neighbors(
                    sensor_positions_list[i],
                    sensor_times_list[i],
                    sensor_positions_list[j],
                    sensor_times_list[j],
                    spatial_threshold,
                    time_threshold
                )

                _neighbor_pairs_set = {
                    (i, j, pairs[0], pairs[1]) for pairs in _neighbor_pairs_set_two_sensors
                }

                neighbor_pairs_set.update(_neighbor_pairs_set)

        return neighbor_pairs_set

    def _find_isotropic_neighbors(self, data_one, data_two, r):
        data_one = np.array(data_one)
        data_two = np.array(data_two)

        if data_one.ndim == 1:
            assert data_two.ndim == 1
            data_one = data_one.reshape(-1, 1)
            data_two = data_two.reshape(-1, 1)

        data_one_tree = cKDTree(data_one)
        data_two_tree = cKDTree(data_two)

        idx_to_neighbors = data_one_tree.query_ball_tree(data_two_tree, r)
        pairs_list = set()
        for i in range(len(idx_to_neighbors)):
            for j in idx_to_neighbors[i]:
                pairs_list.add((i, j))

        return pairs_list

    # def get_number_neighbors(self, sensor_positions_list, sensor_times_list, spatial_threshold, time_threshold):
    #     assert len(sensor_positions_list) == len(sensor_times_list), (len(sensor_positions_list), len(sensor_times_list))
    #
    #     n_sensors = len(sensor_positions_list)
    #
    #     cpt = 0
    #     for i in range(n_sensors):
    #         for j in range(n_sensors):
    #             if i <= j:
    #                 continue
    #
    #             cpt += len(self.find_neighbors(
    #                 sensor_positions_list[i],
    #                 sensor_times_list[i],
    #                 sensor_positions_list[j],
    #                 sensor_times_list[j],
    #                 spatial_threshold,
    #                 time_threshold
    #             ))
    #
    #     return cpt

    def on_received_message(self, message):
        """
        User-defined method and is triggered to handle the message passed by Input.

        Parameters
        ----------
        message : Dictionary
            The message received is in form {'from':agent_name, 'data': data,
            'senderType': agent_class, 'channel':channel_name}. agent_name is the
            name of the Input agent which sent the message data is the actual content
            of the message.

        Assume that data is a dict: {
            "sensor_position": list of size-2 tuples of floats,
            "sensor_time": list of floats,
            "other_sensor_position": list of size-2 tuples of floats,
            "other_sensor_time": list of floats
        }

        Find pairs of indices (i, j) such that the i-th measurement from the sensor is a rendezvous point with the j-th
        measurement from the other sensor.
        """

        data = message["data"]

        sensor_position = data["sensor_position"]
        sensor_time = data["sensor_time"]

        assert len(sensor_position) == len(sensor_time), "Incompatible sizes: {} vs {}".format(
            len(sensor_position), len(sensor_time)
        )

        other_sensor_position = data["other_sensor_position"]
        other_sensor_time = data["other_sensor_time"]

        assert len(other_sensor_position) == len(other_sensor_time), "Incompatible sizes: {} vs {}".format(
            len(other_sensor_position), len(other_sensor_time)
        )

        self.rendezvous_steps_list_buffer = self.find_neighbors(
            sensor_position,
            sensor_time,
            other_sensor_position,
            other_sensor_time,
            self.spatial_threshold,
            self.time_threshold
        )

    def get_random_position_and_time_distance(self, sensor_positions_list, sensor_times_list):
        n_sensors = len(sensor_positions_list)

        _random_idx_sensor = np.random.choice(n_sensors)

        _random_i, _random_j = np.random.choice(len(sensor_positions_list[_random_idx_sensor]), 2, replace=False)

        _random_position_i = sensor_positions_list[_random_idx_sensor][_random_i]
        _random_position_j = sensor_positions_list[_random_idx_sensor][_random_j]
        _random_position_distance = self.get_spatial_distance(_random_position_i, _random_position_j)

        _random_time_i = sensor_times_list[_random_idx_sensor][_random_i]
        _random_time_j = sensor_times_list[_random_idx_sensor][_random_j]
        _random_time_distance = self.get_time_distance(_random_time_i, _random_time_j)

        return _random_position_distance, _random_time_distance

    def select_threshold(
            self,
            sensor_positions_list,
            sensor_times_list,
            sensor_measurements_list,
            n_rep_random=1000,
            min_number_neighbors=20,
            grid_size=10
    ):
        """
        Select spatial and time thresholds for rendezvous points.

        Parameters
        ----------
        sensor_positions_list : list of lists of size-2 tuples of floats
            Coordinates (tuples of float) of measurements of all sensors.
        sensor_times_list : list of lists of floats
            Times (float) of measurements of all sensors.
        sensor_measurements_list : list of lists of floats
            Measurements from all sensors.
        n_rep_random : int
            Number of repetitions to compute medians which will serve for the center of the box of thresholds.
        min_number_neighbors : int
            Minimal number of neighbors to compute a correlation. TODO(): Use a Bayesian estimate.
        grid_size : int
            Correlation optimized on a grid of size grid_size x grid_size.

        The thresholds are chosen by optimizing the correlation between the measurements from rendezvous points.
        TODO:() Use a Bayesian estimator instead of the parameter min_number_neighbors.
        """

        assert len(sensor_positions_list) == len(sensor_times_list), (len(sensor_positions_list), len(sensor_times_list))

        distances_sample = [
            self.get_random_position_and_time_distance(sensor_positions_list, sensor_times_list)
            for _ in range(n_rep_random)
        ]

        # TODO:() Improve or expose quantiles
        quantiles = np.quantile(distances_sample, [0.01, 0.5, 0.99], axis=0)

        spatial_threshold_bounds = [quantiles[0, 0], quantiles[2, 0]]
        times_threshold_bounds = [quantiles[0, 1], quantiles[2, 1]]

        # TODO:() Use log-grids.
        spatial_threshold_grid = np.linspace(spatial_threshold_bounds[0], spatial_threshold_bounds[1], grid_size)
        time_threshold_grid = np.linspace(times_threshold_bounds[0], times_threshold_bounds[1], grid_size)

        correlations = np.zeros([grid_size, grid_size])

        for i in range(grid_size):
            for j in range(grid_size):
                all_neighbors = self.find_all_neighbors(sensor_positions_list, sensor_times_list, spatial_threshold_grid[i], time_threshold_grid[j])
                if len(all_neighbors) <= min_number_neighbors:
                    # Improve convention?
                    correlations[i, j] = 0.0
                else:
                    data_i = []
                    data_j = []

                    for neighbors in all_neighbors:
                        data_i.append(
                            sensor_measurements_list[neighbors[0]][neighbors[2]]
                        )

                        data_j.append(
                            sensor_measurements_list[neighbors[1]][neighbors[3]]
                        )

                    data_i = np.array(data_i)
                    data_j = np.array(data_j)

                    # TODO:() Smooth the data? Use a Bayesian estimate?
                    correlations[i, j] = np.corrcoef(data_i.reshape(1, -1), data_j.reshape(1, -1))[0, 1]

        i_opt, j_opt = correlations.max(1).argmax(), correlations.max(0).argmax()
        spatial_threshold_opt = spatial_threshold_grid[i_opt]
        time_threshold_opt = time_threshold_grid[j_opt]

        self.spatial_threshold = spatial_threshold_opt
        self.time_threshold = time_threshold_opt
