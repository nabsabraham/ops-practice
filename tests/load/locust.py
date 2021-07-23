import time
from locust import HttpUser, task, between
import json

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)
    payload = {
        'question': "Who has been working hard for hugginface/transformers lately?", 
        'context': "Manuel Romero has been working hardly in the repository hugginface/transformers lately"
        }
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}  

    @task
    def index_page(self):
        self.client.post(url="/predictions/qa", json=json.dumps(self.payload))
