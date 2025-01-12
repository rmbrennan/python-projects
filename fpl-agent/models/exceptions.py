# fantasy_pl/models/exceptions.py
class DataFetchError(Exception):
    """Raised when there's an error fetching data from the FPL API"""
    pass