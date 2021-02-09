.ONESHELL:
.PHONY: text
text: 
	python process_text_annotations.py --json_annot_file ../data/CELEBA/raw10.json --out_file processed_annotations/processed_text10.pkl
.PHONY: train
train: 
	. ~/.bashrc
	conda activate t2f
	python train_network.py --config=configs/$(num).conf
