This is a guide to prepare a code environment for machine learning and deep learning with:
WSL + Docker + VSCode + jupyter extension + nvidia GPU + virtual environment (env)

I made this guide because I didn't find these steps in one place, and many tutorials have a different setup or some are not updated. I hope this to be useful for someone else too.
# Windows Subsystem for Linux (WSL)
The WSL allows to install a Linux layer on Windows and work as we were on a Linux environment. It works with virtualization but is fully integrated with all the hardware in our machine.

## Windows Features
Enable windows features and restart the system
- Virtual Machine Platform
- Windows Subsytem for Linux

	<img src="AIEnnvironment Setup/wsl_windows_features.png" style='max-width: 350px;'/>

## WSL Installation

1) To check if WSL is already installed from a command prompt:
```shell
wsl
```

<img src="wsl_check_installation.png" style="max-width:700px;"/>
	Here there is 3 possibilities:
	- Not installed
	- Installed without a distro
	- Installed with a distro

### WSL not installed

1) To install the WSL we execute the command:
```shell
#If we want to install the default distro Ubuntu:
wsl --install

#if we want to install directly a specific distro we use:
wsl --install <Distro>
```
<img src="wsl_install_1.png" style="max-width:700px;"/>
2) In the process it will ask to create a root user:
	UNIX username: davidayala
	New password: LearningPath

<img src="wsl_install_2.png" style="max-width:700px;"/>
2) When the installation has finished, it will automatically log us into the Linux terminal, we can quit by using the command "exit"

3) Is recommended to always do the following steps:

```shell
#we check the installation with:
#wsl

#Is always a good idea to update the wsl
#wsl --update
```

### WSL installed without Distro

1) First, we need to update the WSL with the command:
```shell
wsl --update
```

2) We can install Ubuntu but if we want to check other Distros or a specific version, we can execute the following command to check the available Distros:
```shell
wsl —list —online
```

3) We have to install the distro by using the command:
```shell
#We can use any other distro or specific version also
wsl --install -d Ubuntu
```
## Turn on virtualization
To verify if the virtualization is enabled we can check the status on the task manager

<img src="wsl_enable_virtualization.png" style="max-width:600px;"/>

## WSL Distro Terminal Access

1) We can list all the distros installed in our WSL by using the following command in the command prompt:
```shell
wsl --list --verbose
```
![[wsl_distro_1.png]]


2) To access the terminal of our distro we can use the name of the distro
![[wsl_distro_2.png]]

3) We can quit by using the command:
```shell
exit
```
![[wsl_distro_3.png]]



# Docker



Video Guide: **[https://www.youtube.com/watch?v=3JU7Pjwk4s0](https://www.youtube.com/watch?v=3JU7Pjwk4s0)**

## GUI Installation

1) Download Desktop version
<img src="docker_gui_installation_1.png"  style="max-width:500px;"/>

2) Execute the installer

<img src="docker_gui_installation_2.png"  style="max-width:300px;"/>

3) Setup docker installation


<img src="docker_gui_installation_3.png"  style="max-width:400px;"/>

4) You can skip to login with an account but is better to do it (we also can login from the command prompt)
<img src="docker_gui_installation_4.png"  style="max-width:500px;"/>

5) You can skip all the other steps and you should see the following

<img src="docker_gui_installation_5.png"  style="max-width:500px;"/>

6) In settings we need to enable the WSL integration for the Linux distro (this is a requirement to enable the use of GPU in our code)

<img src="docker_gui_installation_6.png"  style="max-width:500px;"/>

## Verify Installation

From the **==command prompt==** we can do the following:

1) You can check the installed version with this command:
```shell
docker --version
```

2) To verify the correct functionality of docker we can execute the following command:
   This command will pull a docker image, create a container from the image and access the container
```shell
docker run hello-world
```

<img src="docker_verification_1.png" style="max-width:650px;"/>


## Prompt Login

If we need to login to docker from the command prompt without and account we can use:
This will open the login page on the browser.
```shell
docker login docker.io
```

<img src="docker_prompt_login.png" style="max-width:650px;"/>

But if we have an account we can use directly:
```shell
docker login -u our_username
```


# NVIDIA Container Toolkit

## Toolkit Installation

Before follow these steps the recommendation is to look for this information in the official page NVIDIA. You can find the information by searching "**Installing the NVIDIA Container Toolkit**".

Also you could go directly by using this link:
**[https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**

All these steps have to be entered in our **==Linux terminal==**.

1) Add the gpg  key for the repo
```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
<img src="wsl_setup_gpu_1.png" style="max-width:650px;"/>

2) Add the repo into our sources.list.d
```shell
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

<img src="wsl_setup_gpu_2.png" style="max-width:650px;"/>

3) Update the apt package
```shell
sudo apt-get update
```
<img src="wsl_setup_gpu_3.png" style="max-width:650px;"/>

4) Install nvidia toolkit
```shell
sudo apt-get install -y nvidia-container-toolkit
```
<img src="wsl_setup_gpu_4.png" style="max-width:650px;"/>

5) Docker configuration
```shell
sudo nvidia-ctk runtime configure --runtime=docker
```
<img src="wsl_setup_gpu_5.png" style="max-width:650px;"/>

6) Now restart the complete system.
   In some places we can find to run the command "sudo systemctl restart docker", but as we are using a WSL we need to restart the full system

## Installation Verification

1) To verify the installation we can use this command on the Linux terminal:
```shell
docker run --rm --gpus all ubuntu nvidia-smi
#In older nvidia versions you will need to use --runtime=nvidia, but in current versions --gpus all do this automatically
#docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```
This command will:
- ubuntu: Download the image ubuntu
- run: Create a container based on the image and access to it
- --gpus all: Allow use of the GPU to the container
- nvidia-smi: Run the nvidia-smi to check the GPU
- --rm: Delete the container

2) If we get in return a table with the information of the GPU it means it worked correctly

<img src="wsl_setup_gpu_verify_1.png" style="max-width:650px;"/>

3) Optionally if we want to delete the image created we first need to check the name or id of the image and then delete it with the following commands:
```shell
#This will list the created images and its name and id
docker images
```
<img src="wsl_setup_gpu_verify_2.png" style="max-width:650px;"/>

```shell
#We can delete the image by its id
docker rmi <image_id>
#or que can use the image name
docker image rm <image_name>
```

# Visual Studio Code

## Visual Studio Code Installation
This is a simple installation of Visual Studio Code with all the values by default.

<img src="vscode_installation_1.png"  style="max-width:400px;"/>


<img src="vscode_installation_2.png"  style="max-width:400px;"/>


<img src="vscode_installation_3.png"  style="max-width:400px;"/>

## Visual Studio Code Extensions

1) Search and install the extension Dev Containers

<img src="vscode_ext_devcontainers.png"  style="max-width:650px;"/>

2) Search and install the extension Docker

<img src="vscode_ext_docker.png"  style="max-width:650px;"/>

3) Search and install the extension Python Extension


<img src="vscode_ext_python.png"  style="max-width:650px;"/>


4) Search and install the extension Jupyter

<img src="vscode_ext_jupyter.png"  style="max-width:650px;"/>


## Visual Studio Code Workspace


1) We need to define/select a directory as the workspace for our projects, for this we open a folder and mark it as a trusted directory

<img src="vscode_setup_01.png"  style="max-width:650px;"/>


<img src="vscode_setup_02.png"  style="max-width:400px;"/>


<img src="vscode_setup_03.png"  style="max-width:350px;"/>

2) Show the terminal window

<img src="vscode_setup_04.png"  style="max-width:650px;"/>

3) On the terminal we can work as if we will do it in the command prompt, so we will access to our Linux distro

<img src="vscode_setup_05.png"  style="max-width:650px;"/>

4) From here we have two options:
   - Create a docker image with all the setup we will need for our container, so if we need a new container we will just created from the image and we won't need to do anything else.
   - Create a container and do all the setup on it.
   
	On this guide I'll use the second option but if you want to use the first option, you just need to check how to create the docker file and include many of the commands reviewed in the section "Container Setup/Manual Setup".


## Container Creation with NVIDIA/CUDA

The easiest option to have a clean container is to download a publica image created by NVIDIA with all the setup we need to use the GPU. From this image we will create the container and install everything, including python.


1) We need to pull (download) the NVIDIA image, which we can check for the image name in the docker hub page: https://hub.docker.com/search?q=nvidia
2) Once we have it we use the command:
```shell
#This command will download the image, but sometimes there are connection problems
docker pull nvidia/cuda:12.6.3-base-ubuntu24.04

#So if the previous command does not work, we can use this code that will create a container and immediatly after will delete it
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```
3) We will see the image on the docker panel
<img src="vscode_container_creation_01.png"  style="max-width:650px;"/>

4) Now that we have the image we will create the container
```shell
docker run --gpus all -it --name container_name -v ${pwd}:/app nvidia/cuda:12.6.3-base-ubuntu24.04
```
	This command will do the following:
	- run: creates the container
	- --gpus all: enable the use of the GPU
	- -it: enable the possibility to make changes to the container
	- --name container_name: we can assign a custom name to the container
	- ${pwd}:/app: this will create the container in the current path

5) After create the container we will see the container created in the docker panel
<img src="vscode_container_creation_02.png"  style="max-width:650px;"/>

## Container Window

To work with the container on Visual Studio Code, we will open it in a new window with the following steps.

1) Right click on the container and select the option "Attach Visual Studio Code"
<img src="vscode_container_detach_01.png"  style="max-width:650px;"/>

2) Allow to trust the container
<img src="vscode_container_detach_02.png"  style="max-width:300px;"/>

3) In the new window, we will open the folder of the workspace

<img src="vscode_container_detach_03.png"  style="max-width:650px;"/>


4) We need to select the /app/ path which is associated with the workspace due to the command we used to create the container.

<img src="vscode_container_detach_04.png"  style="max-width:650px;"/>


5) Also we will need to open the terminal window (also in the container window) to continue with the next section.

# Container Setup

To make the setup of the container we have to options:
- Manual Setup, with this option we will manually execute each command on the terminal of the container window.
- Batch File Setup, with this option we will have a file prepared with all the commands we need for our container, so when we need to create a new container we will just execute the file.
## Manual Setup

All these commands have to be executed on the ==**container terminal**==.
### 1) Python Installation

1) First we need to update the apt package
```python
apt update
```

2) We need to install python with the following command:
```python
apt install -y python3 python3-pip python3-venv
```
### 2) Virtual Environment Creation

1) Create a virtual environment (venv), with this command we will create the virtual environment in the given path /env
```python
python3 -m venv /env
```

2) We activate the virtual environment to install new packages on it
```python
source /env/bin/activate
```


### 3) Jupyter Installation

1) We install the Jupyter package, we nee to use both commands
```python
# Jupyter installation
pip3 install jupyter
pip install ipykernel
```

2) We need to create a kernel to access the created virtual environment. When we create a Jupyter notebook we will need to select the kernel to run our code.
   Here the "myenv" corresponds to the name we give to our kernel
```python
# (reboot the host environment after this)
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
```


## Batch File Setup

We can also create a file with all the commands to install the packages that we will need in our container.

1) We have to create a file with the commands with a name and extension like "container_setup.sh". This is an example of the content of the file.

```python
set -e # Exit immediately on error

# Update package lists
# --------------------------------------------------------------------------------
apt update

# Install Python
# --------------------------------------------------------------------------------
apt install -y python3 python3-pip python3-venv

# Create Virtual Environment on the container
# --------------------------------------------------------------------------------
# Create Virtual environment (mandatory for security policies since Python 3.12)
ENV_NAME="myenv"
ENV_PATH="/env"

python3 -m venv $ENV_PATH #This is used to create the virtual environment in the given path
source $ENV_PATH/bin/activate #This activate the venv for all the other commands to run on the venv

# Install common OS packages
# --------------------------------------------------------------------------------
apt-get update && apt-get install -y git curl nano

pip install --upgrade pip

# Install AI Packages
# --------------------------------------------------------------------------------

# Jupyter installation
pip3 install --no-cache-dir jupyter
pip install --no-cache-dir ipykernel 

# Install common data science and AI packages
pip install --no-cache-dir numpy pandas matplotlib seaborn
pip install --no-cache-dir datasets
pip install --no-cache-dir mlxtend charset_normalizer fuzzywuzzy
pip install --no-cache-dir unidecode

# Natural Language Processing
pip install --no-cache-dir nltk
pip install --no-cache-dir gensim

# Install AI models and tools
pip install -U scikit-learn
pip install xgboost
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Create kernel for python venv to show it in the kernel selection on the jupyter notebook
# (reboot the host environment after this)
python -m ipykernel install --user --name=$ENV_NAME --display-name "Python ($ENV_NAME)"
```

2) We can save the file on the root directory of our container
<img src="vscode_container_batch_setup_01.png"  style="max-width:650px;"/>

3) Also we need to give the file read and write permission with the following command:
```shell
chmod +x container_setup.sh
```

4) To execute our file we use the command:
```shell
bash container_setup.sh
```

