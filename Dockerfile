##
##      To build the AI_GFPGAN docker image
##

# base stuff
#FROM yeepeekoo/public:ai_base_osais
FROM yeepeekoo/public:ai_facex

# install requirements
#RUN pip3 install facexlib
RUN pip3 install realesrgan
RUN pip3 install basicsr
RUN pip3 install \
    lmdb \
    numpy \ 
    opencv-python \
    pyyaml \
    scipy \
    tb-nightly \
    torch>=1.7 \
    torchvision \
    tqdm \
    yapf

# keep ai in its directory
RUN mkdir -p ./ai
RUN chown -R root:root ./ai
COPY ./ai ./ai

# latest of osais base
COPY ./_temp ./

# copy OSAIS -> AI
COPY ./gfpgan ./gfpgan
COPY ./gfpgan.json .

# overload config with those default settings
ENV ENGINE=gfpgan

# run as a server
CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "5012"]
