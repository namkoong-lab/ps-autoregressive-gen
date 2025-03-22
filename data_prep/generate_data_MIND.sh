# replace with yours
MIND_DATA_DIR=/shared/share_mala/implicitbayes/dataset_files/MIND_data/

curl --create-dirs -O --output-dir $MIND_DATA_DIR/large/train 'https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip' 
unzip $MIND_DATA_DIR/large/train/MINDlarge_train.zip -d $MIND_DATA_DIR/large/train/
python process_MIND.py --data_dir $MIND_DATA_DIR/large/ 
python process_MIND_more.py --data_dir $MIND_DATA_DIR/large/ --save_dir $MIND_DATA_DIR/filter100/
