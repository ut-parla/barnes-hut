.PHONY: all

all:
	@echo sudo apt install -y python3.7 python3.7-dev python3.7-venv
	@echo python3.7 -m venv .parla
	@echo . .parla/bin/activate 
	@echo pip install wheel && pip install -r requirements.txt
	@echo git submodule init
	@echo pip install -e Parla.py 

	@echo - If numba installation fails.. gere comes the pain..installing numba required llvmlite, which
	@echo requires llvm, and for some reason everytime I tried installing
	@echo numba, it complained about llvm-config.
	@echo First, please check if $llvm-config exists.
	@echo If not, install it:
	@echo   sudo apt install llvm-9
	@echo then export the path:
	@echo   export PATH="${PATH}:/usr/lib/llvm-9/bin"
	@echo then finally install numba
	@echo   . .parla/bin/activate \&\& pip install numba 

run:
	@echo . .parla/bin/activate 
	@echo cd python-bh
	@echo export LD_LIBRARY_PATH=/usr/local/cuda/lib64
	@echo ./bin/run_2d.py input/n10k.txt 1 1 configs/default.ini

