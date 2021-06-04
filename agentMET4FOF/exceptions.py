class SensorsNotLinearlyIndependentError(Exception):
    """
    Custom exception to handle the case when sensor results are not linearly independent
    """
    pass


class SystemMatrixNotReducibleError(Exception):
    """
    Custom exception to handle the case when the system matrix *A* is not reducible
    """
    pass


class ColumnNotZeroError(Exception):
    """
    Custom exception to handle the case when a redundant column has not been reduced to zero
    """
    pass