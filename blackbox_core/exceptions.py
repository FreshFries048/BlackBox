"""
BlackBox Core Exceptions
"""

class BlackBoxError(Exception):
    """Base exception for BlackBox system."""
    pass

class MissingFeatureError(BlackBoxError):
    """Raised when required data features are missing from DataFrame."""
    
    def __init__(self, node_name: str, metric: str):
        self.node_name = node_name
        self.metric = metric
        super().__init__(f"Node '{node_name}' requires metric '{metric}' which is missing from data")
