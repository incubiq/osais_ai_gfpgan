##
##      To build the AI_GFPGAN_ docker image
##

# base stuff
FROM yeepeekoo/public:ai_facex

# install requirements
RUN pip3 install realesrgan==0.3.0
RUN pip3 install basicsr==1.4.2
RUN pip3 install \
    lmdb==1.4.1 \
    numpy==1.23.5 \ 
    opencv-python==4.7.0.72 \
    pyyaml==6.0 \
    scipy==1.10.1 \
    tb-nightly==2.14.0a20230625 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    tqdm==4.65.0 \
    yapf==0.40.1


# keep ai in its directory
RUN mkdir -p ./ai
RUN chown -R root:root ./ai
COPY ./ai ./ai

# copy OSAIS -> AI
COPY ./gfpgan ./gfpgan
COPY ./gfpgan.json .
COPY ./_gfpgan.py .

# overload config with those default settings
ENV ENGINE=gfpgan

# run as a server
CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "5012"]
