class SensorsNotLinearlyIndependentError(Exception):
    """Custom exception to indicate linearly dependent sensor results"""
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
