
now=$(date +"%T")
echo "Current time : $now"

$echo curl -X POST https://127.0.01:5000/predictions/my_tc -T sample2.txt --insecure

$echo curl -X POST https://127.0.01:5000/predictions/my_tc -T sample.txt --insecure
now=$(date +"%T")
echo "Current time : $now"
