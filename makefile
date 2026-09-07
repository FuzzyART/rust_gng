.PHONY: up build clean rebuild

APP_DIR := my_app
NIX_CMD := nix develop --command bash -c

up:
	./scripts/container/2-startContainer.sh
	./scripts/container/3-install_gng.sh

build_cont:
	./scripts/container/1-buildContainer.sh
	./scripts/container/3-install_gng.sh

remove_cont:
	docker rm -f pytorch_project_cont

clean:
	./scripts/5-clean_lib.sh


build_lib:
	./scripts/1-build_lib.sh

build_lib_release:
	./scripts/2-build_py_lib.sh



#clean:
	#$(NIX_CMD) "cd $(APP_DIR) && cargo clean"

