import requests, datetime

class Selic():
    def __init__(self):
        self.rate = self._get_request()
        self._url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados"
        self._params = {
                "formato": "json",
                "dataInicial": datetime.date.today().strftime("%d/%m/%Y"),
                "dataFinal": datetime.date.today().strftime("%d/%m/%Y")
                }

    def _get_request(self)-> float:
        try:
            response = requests.get(self.url, params=self.params)
            if response.status_code == 200:
                selic_today = response.json()
                return float(selic_today[0]['valor']) / 100
        except Exception as e:
            print(f"Erro: {response.status_code}")