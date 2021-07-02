class SensorsNotLinearlyIndependentError(Exception):
    """Custom exception to indicate linearly dependent sensor results"""
    pass


class SystemMatrixNotReducibleError(Exception):
    """Custom exception to handle non-reducible system matrix *A*"""
    pass


class ColumnNotZeroError(Exception):
    """Custom exception to handle redundant columns not reduced to zero"""
    pass
