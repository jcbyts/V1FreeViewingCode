
declare -a arr=("20191206_kilowf")
                # "20191205_kilowf"
                # "20191119_kilowf"
                # "20191120a_kilowf"
                # "20191121_kilowf"
                # "20191122_kilowf"
                # "20191231_kilowf"
                # "20200304_kilowf")

for i in "${arr[@]}"
do 
    echo "Fitting session $i"
    python v1_tracker_calibration.py --name=$i
done