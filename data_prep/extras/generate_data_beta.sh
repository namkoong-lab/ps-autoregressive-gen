# REPLACE WITH YOURS
datadir=/shared/share_mala/implicitbayes/dataset_files/synthetic_data/beta

N=500
D_eval=10000
D=25000
cnts=5
python generate_beta_data.py --cnts $cnts --datadir $datadir --D $D --N $N --D_eval $D_eval

