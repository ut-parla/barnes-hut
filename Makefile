all:
	sudo apt install -y python3.7 python3.7-dev python3.7-venv
	. .bh/bin/activate && pip install wheel && pip install -r requirements.txt
	rm -rf Parla.py
	git clone git@github.com:ut-parla/Parla.py.git
	. .bh/bin/activate && cd Parla.py && python3 setup.py install

	@echo - Here comes the pain..installing numba required llvmlite, which
	@echo requires llvm, and for some reason everytime I tried installing
	@echo numba, it complained about llvm-config.
	@echo First, please check if $llvm-config exists.
	@echo If not, install it:
	@echo   sudo apt install llvm-9
	@echo then export the path:
	@echo   export PATH="${PATH}:/usr/lib/llvm-9/bin"
	@echo then finally install numba
	@echo   . .bh/bin/activate \&\& pip install numba 
