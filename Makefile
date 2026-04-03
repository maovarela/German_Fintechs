IMAGE_NAME  ?= german-fintech-ml
DATA_VOL    ?= $(CURDIR)/data
MLRUNS_VOL  ?= $(CURDIR)/mlruns

# Set KAGGLE_API_TOKEN in your shell before running make train:
#   export KAGGLE_API_TOKEN=KGAT_xxx

.PHONY: build train serve clean

build:
	docker build -t $(IMAGE_NAME) .

train: build
	docker run --rm \
		-e MODE=train \
		-e KAGGLE_API_TOKEN="$(KAGGLE_API_TOKEN)" \
		-v "$(DATA_VOL):/app/data" \
		-v "$(MLRUNS_VOL):/app/mlruns" \
		$(IMAGE_NAME)

serve:
	docker run --rm \
		-e MODE=serve \
		-v "$(DATA_VOL):/app/data" \
		-v "$(MLRUNS_VOL):/app/mlruns" \
		-p 8000:8000 \
		$(IMAGE_NAME)

clean:
	docker rmi -f $(IMAGE_NAME) 2>/dev/null || true
	rm -rf data mlruns
