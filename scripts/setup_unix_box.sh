# install python
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.9
sudo apt install -y python3.9-distutils

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py

echo 'alias python=python3.9' >> .bash_profile
echo "alias pip='python3.9 -m pip'" >> .bash_profile
echo "PATH=/home/spangher/.local/bin:$PATH" >> .bash_profile
source .bash_profile


pip install playwright==1.30
pip install pandas
pip install pytimeparse
pip install matplotlib
pip install tqdm
pip install more_itertools
pip install scipy
pip install beautifulsoup4
pip install --upgrade setuptools
pip install waybackpack

sudo apt-get install tmux
playwright install
playwright install chrome


# to set up singlefile docker
sudo snap install docker
sudo docker pull capsulecode/singlefile
sudo docker tag capsulecode/singlefile singlefile
sudo docker login


# to install single-file-cli manually (doesn't yet work on unix)
sudo snap install chromium
sudo apt install npm

#sudo su
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
nvm install 17
npm install -g single-file-cli
npm install -g puppeteer
sudo apt-get update
sudo apt-get install chromium-browser



# single-file \
#     --urls-file=url-list.txt \
#     --block-images \
#     --browser-executable-path=/usr/bin/chromium-browser \
#     --max-parallel-workers 1 \
#     --output-directory \


