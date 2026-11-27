"""The contract every NLP feature follows."""
from abc import ABC, abstractmethod


class NLPModule(ABC):
    """Subclass this to add a feature.

    name        URL slug: the feature is served at POST /api/<name>
    title       label shown in the UI
    description one line shown under the title
    """

    name: str
    title: str
    description: str
    order: int = 100  # position of the tab in the UI

    @abstractmethod
    def run(self, text: str) -> dict:
        """Analyse text and return a JSON-serialisable dict.

        Raise ValueError for input the user can fix.
        """
