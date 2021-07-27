
declare -a arr=("gru20210525")


for i in "${arr[@]}"
do 
    echo "Fitting session $i"
    python v1_tracker_calibration.py --name=$i --stimlist=["Gabor", "BackImage", "DriftingGrating"]
done