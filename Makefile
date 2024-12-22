.PHONY: train run gt clean

train:
	python train_flame_voca.py
run:
	python run_flame_voca.py
gt:
	python render_gt.py
clean:
	rm -rf outputs/
	rm -rf models/
	rm -rf checkpoints/
	rm -rf logs/
