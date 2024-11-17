FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
RUN apt update && apt install python3-pip git vim sudo curl wget apt-transport-https ca-certificates gnupg libgl1 -y
RUN pip install setuptools
RUN pip3 install pandas numpy scipy six wheel jax[cuda]==0.4.16 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jaxlib==0.4.16 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install notebook matplotlib tqdm jupyter ipython wandb rich
RUN pip3 install distrax==0.1.5 brax==0.9.2 chex==0.1.8 flax==0.6.11 optax==0.1.7 orbax==0.1.9 jaxopt==0.8.1 gym==0.26.2  gymnax==0.0.6 mujoco==2.3.7 tensorflow-probability==0.22.0 scipy==1.11.3
RUN echo 'export PATH=$PATH:/home/duser/.local/bin' >> ~/.bashrc
RUN wandb login 4660672e2c993e2d293d3153ca4ddc82f91a4736
RUN apt-get update && apt-get install -y p7zip-full
RUN apt-get update && apt-get install -y unrar
RUN apt-get update && apt-get install -y htop
ARG UID

RUN useradd -u $UID --create-home duser && \
   echo "duser:duser" | chpasswd && \
   adduser duser sudo && \
   mkdir -p /home/duser/.local/bin && \
   chown -R duser:duser /home/duser

USER duser
WORKDIR /home/duser/
RUN git config --global user.email "reuben@robots.ox.ac.uk"
RUN git config --global user.name "reuben"
ARG GIT_TOKEN

# Switch to non-root user
USER duser
RUN echo "alias i='/usr/local/bin/ipython'" >> ~/.bashrc
