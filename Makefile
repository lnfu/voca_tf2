.PHONY: train run clean

train:
	python train_flame_voca.py
run:
	python run_flame_voca.py
clean:
	rm -rf models/
	rm -rf checkpoints/
	rm -rf outputs/
	rm -rf logs/