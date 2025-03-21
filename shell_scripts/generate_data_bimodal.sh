# replace with yours
datadir=/shared/share_mala/implicitbayes/dataset_files/synthetic_data/bimodal

# data gen params
N=500
D_eval=1000
D=2500
cnts1=25
cnts2=25

python generate_bimodal_data.py --cnts1 $cnts1 --cnts2 $cnts2 --datadir $datadir --D $D --N $N --D_eval $D_eval 
