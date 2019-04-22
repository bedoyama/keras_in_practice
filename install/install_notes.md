1. Theano

pip install theano
conda install m2w64-toolchain
conda install -c anaconda libpython

https://stackoverflow.com/questions/50793797/importerror-version-check-of-the-existing-lazylinker-compiled-file-looking-for

2. keras on Docker

2.1. Get Docker

Windows 10 https://docs.docker.com/docker-for-windows/install/
Windows 7 https://docs.docker.com/toolbox/overview/

2.2. git clone https://github.com/floydhub/dl-docker.git

docker build -t floydhub/dl-docker

2.3. git clone https://github.com/saiprashanths/dl-docker.git

RUN Docker Quickstart Terminal

"c:\Program Files\Docker Toolbox\docker-machine.exe" env default

use output to create setDocketbat and run it

docker build -t floydhub/dl-docker:gpu -f Dockerfile.gpu .

Get coffee 

docker run -it -p 8888:8888 -p 6066:6066 floydhub/dl-docker:gpu bash