#ARG BASE_IMAGE=rapidsai/rapidsai:22.06-cuda11.0-runtime-ubuntu18.04-py3.8
ARG BASE_IMAGE=continuumio/miniconda3:latest

FROM ${BASE_IMAGE}

# # disable terminal interaction for apt
# ENV DEBIAN_FRONTEND=noninteractive
# ENV LC_ALL=C

# Get user parameters from runner
ARG USERNAME
ARG USER_UID
ARG USER_GID
ARG CONDA_ENV_NAME

#Check args reception
RUN echo "Creation" \
    echo "USERNAME = $USERNAME" \
    echo "GID = $USER_GID" \
    echo "UID = $USER_UID" \
    echo "CONDA_ENV_NAME = $CONDA_ENV_NAME"

# Set user parameters as environment variables
ENV USERNAME=${USERNAME}
ENV USER_GID=${USER_GID}
ENV USER_UID=${USER_UID}

# USER CREATION
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.

# PACKAGES  INSTALL
RUN apt-get update \
    && apt-get install ffmpeg libsm6 libxext6  -y \
    && apt-get install wget git -y \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# # NVTOP INSTALL
# RUN cd /tmp \
#     && apt install -y build-essential \
#     && apt install -y cmake libncurses5-dev libncursesw5-dev git \
#     && git clone https://github.com/Syllo/nvtop.git \
#     && mkdir -p nvtop/build && cd nvtop/build \
#     && cmake .. -DNVIDIA_SUPPORT=ON -DAMDGPU_SUPPORT=ON \
#     && make install

# # HTOP
# RUN apt install htop

RUN cd / && mkdir python_packages

# Log as $USER and create a .bashrc file for config
USER ${USER_UID}:${USER_GID}


# # GENERAL PURPOSE INSTALLS
# # RUN sudo apt-get update && sudo apt-get install -y \
# #     bash-completion locales \
# #     && sudo rm -rf /var/lib/apt/lists/*

# # LOCALES GIT BUG PREVENTION
# # RUN locale-gen en_US.UTF-8
# # RUN sudo localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

# Add possibility to write in /opt
RUN sudo chown -R $USER_UID:$USER_GID /opt/conda/

RUN touch ~/.bashrc
# # RUN sudo mkdir /opt/conda/pkgs
# # RUN sudo touch /opt/conda/pkgs/urls.txt

# # Install miniconda
# #ENV CONDA_DIR ~/.conda
# ENV CONDA_DIR /opt/conda
# # Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

COPY environment.yml /tmp/environment.yml

# # PYTORCH: https://stackoverflow.com/questions/71162459/why-does-anaconda-install-pytorch-cpuonly-when-i-install-cuda
# RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc


SHELL ["/bin/bash", "--login", "-c"]

RUN source /home/$USERNAME/.bashrc \
    && conda update conda \
    && conda env update --file /tmp/environment.yml

# #RUN conda create --name $CONDA_ENV_NAME
#RUN echo "conda activate $CONDA_ENV_NAME" >> ~/.bashrc
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate $CONDA_ENV_NAME" >> ~/.bashrc && \
    echo "export PYTHONPATH=\$PYTHONPATH:/python_packages/" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

# ENV PATH ~/.conda/envs/$CONDA_ENV_NAME/bin:$PATH

# # RUN source /home/$USERNAME/.bashrc \
# #     && conda create -q --name $CONDA_ENV_NAME \
# #     && conda activate $CONDA_ENV_NAME \
# #     && conda install -y -c conda-forge pandas xlrd xlwt pytables \
# #     pre-commit black scipy




ENV CONDA_ENV /opt/conda/envs/${CONDA_ENV_NAME}/bin/


SHELL ["/bin/bash", "--login", "-c"]
RUN /bin/bash ~/.bashrc
# ENTRYPOINT ["/bin/bash"]
