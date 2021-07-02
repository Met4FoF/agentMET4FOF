class SensorsNotLinearlyIndependentError(Exception):
    """Custom exception to indicate linearly dependent sensor results"""
    pass


class SystemMatrixNotReducibleError(Exception):
    """Custom exception to handle not reducible system matrix *A*"""
    pass


class ColumnNotZeroError(Exception):
    """Custom exception to handle redundant columns not reduced to zero"""
    pass
