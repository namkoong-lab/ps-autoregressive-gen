# replace with yours
datadir=/shared/share_mala/implicitbayes/dataset_files/synthetic_data/bimodal_smalltest/

# data gen params
N=500
D_eval=1000
D=2500
cnts1=25
cnts2=25
for D in 25 50 100 250 500 1000
do
python ../generate_bimodal_data.py --cnts1 $cnts1 --cnts2 $cnts2 --datadir $datadir --D $D --N $N --D_eval $D_eval
done
