Commands to create the environment 'unet_env':
1. Create the conda environment:
   conda create -n unet_env python=3.8
2. Activate the conda environment:
   conda activate unet_env
3. Install conda packages:
   conda env update --file unet_env_environment.yml --prune
4. Install pip packages:
   pip install -r unet_env_requirements.txt
