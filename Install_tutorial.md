# Create a new conda environment 
conda remove -n fanasr2024 --all   
conda create -n funasr python=3.10   
conda activate funasr

# Install the required packages  
pip3 install -e ./   
pip3 install torch torchvision torchaudio   
pip install hdbscan   
pip install -U rotary_embedding_torch   
pip install ffmpeg-python   cd 
sudo apt-get update   
sudo apt-get install sox   
pip install sox   
pip install tensorboardX   
pip install -U transformers  
pip install lightning
pip install PyYAML
pip install yacs

# Install SOX  
cd /usr/lib/x86_64-linux-gnu/   
sudo cp libsox.so.3 libsox.so   

