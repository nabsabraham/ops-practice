#!/bin/bash

for i in {1..10000};
do
	#(echo -n '{"base64": "'; base64 /home/ubuntu/timg.jpeg; echo '"}') | curl -H "Content-Type: application/json" -d @-  http://localhost:8080/predictions/SpamCls &
	curl -X POST http://127.0.0.1:5000/predictions/qa -T ../data/sample.txt &
    if (( $i % 100 == 0 ))
	then
		sleep 1;
	fi
done;