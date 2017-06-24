
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
##requirements: test_environment
##	conda install -f requirements.yml

## Make Dataset
data:
	bash scripts/load_caltech101.sh 
