IMAGE_NAME := zhaoxuezhi/basemodel_elm
VERSION := 0.2.4

DOCKER_NAMESPACE := harbor.4pd.io/lab-platform/pk_platform
TARGET_CONTAINER_PLATFORMS := linux/amd64

.PHONY: build
build:
	@docker build -t $(IMAGE_NAME):$(VERSION) .

.PHONY: push
push:
	@docker buildx build --push --platform $(TARGET_CONTAINER_PLATFORMS) --tag $(DOCKER_NAMESPACE)/$(IMAGE_NAME):$(VERSION) .
