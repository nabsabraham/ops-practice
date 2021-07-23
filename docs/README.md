### creating custom services with torchserve
https://github.com/pytorch/serve/blob/master/docs/custom_service.md


#### check which models have been registered at endpoint: 
`curl "http://localhost:5001/models"`

#### check the post request
`curl -X POST http://127.0.0.1:5000/predictions/qa -T data/sample.txt`


#### tests

kill the locust server by finding the PID: `lsof -n -i4TCP:8089` `kill -9 PID`