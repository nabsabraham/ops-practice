
now=$(date +"%T")
echo "Current time : $now"

$echo curl -X POST http://127.0.0.1:5000/predictions/qa -T ../data/sample.txt

$echo curl -X POST http://127.0.0.1:5000/predictions/qa -T ../data/sample2.txt
now=$(date +"%T")
echo "Current time : $now"

