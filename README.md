# RoboShouder

## Data points

- data_raw : Recorded points expressed in camera reference frame 
- data_composites : Recorded points expressed in scapula-composite reference frame 
- data_processed : Recorded points expressed in scapula-stl reference frame after a bone morphing step


## Description

This repository contains the data source related to the RoboShouder project : 

- [x] STL files of specimens
- [x] Landmarks positions for all specimens
- [x] Measurements datasets by composite markers
- [x] Processed data from composite markers 
- [x] Python scripts to process data


## Installation 

If you want to check the data and use the scripts via the *devcontainer* tool, you need to install the following items :

- [x] Python 3.7.7 or higher
- [x] Docker (https://docs.docker.com/get-docker/)
  - [x]  For Windows user's, link wsl2 to your Docker installation : https://docs.docker.com/desktop/windows/wsl/
- [x] VScode (https://code.visualstudio.com/download)  
  - [x] VScode extensions :
    - **Docker**


        Id: ms-azuretools.vscode-docker
        Description: Makes it easy to create, manage, and debug containerized applications.
        Version: 1.22.1
        Publisher: Microsoft
        VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker
    - **Remote - Containers**


        Id: ms-vscode-remote.remote-containers
        Description: Open any folder or repository inside a Docker container and take advantage of Visual Studio Code's full feature set.
        Version: 0.241.3
        Publisher: Microsoft
        VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers
    - **Remote Development**


        Id: ms-vscode-remote.vscode-remote-extensionpack
        Description: An extension pack that lets you open any folder in a container, on a remote machine, or in WSL and take advantage of VS Code's full feature set.
        Version: 0.21.0
        Publisher: Microsoft
        VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack
    - **Remote - WSL (For Windows user's)**


        Id: ms-vscode-remote.remote-wsl
        Description: Open any folder in the Windows Subsystem for Linux (WSL) and take advantage of Visual Studio Code's full feature set.
        Version: 0.66.3
        Publisher: Microsoft
        VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl
    - **Python**


        Id: ms-python.python
        Description: IntelliSense (Pylance), Linting, Debugging (multi-threaded, remote), Jupyter Notebooks, code formatting, refactoring, unit tests, and more.
        Version: 2022.10.1
        Publisher: Microsoft
        VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=ms-python.python

## Usage

1 - In VScode, open palette command using the shortcut : *ctrl + shift + p*

2 - Tap on the **Remote-Containers: Open Folder in Container** command

3 - Select the *RoboShouder* folder.

4 - Wait until container builds and starts.
