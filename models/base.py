from abc import ABC, abstractmethod

class OptionPricingModel(ABC):
    """
    Tämä on abstrakti kantaluokka. Se määrittää säännöt,
    joita kaikkien hinnoittelumallien on noudatettava.
    """

    @abstractmethod
    def calculate_price(self, option_type='call'):
        """
        Jokaisen mallin ON pakko toteuttaa tämä funktio.
        Jos malli ei tätä toteuta, koodi ei suostu toimimaan.
        """
        pass
