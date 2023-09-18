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
    Pylance\
    yapf

COPY ./gfpgan ./gfpgan

# keep ai in its directory
RUN mkdir -p ./ai
RUN chown -R root:root ./ai
COPY ./ai/gfpgan ./ai/gfpgan
COPY ./ai/options ./ai/options
COPY ./ai/scripts ./ai/scripts
COPY ./ai/tests ./ai/tests
COPY ./ai/inference_gfpgan.py ./ai/inference_gfpgan.py
COPY ./ai/setup.py ./ai/setup.py
COPY ./ai/setup.cfg ./ai/setup.cfg
COPY ./ai/runai.py ./ai/runai.py

# push again the base files
COPY ./_temp/static/* ./static
COPY ./_temp/templates/* ./templates
COPY ./_temp/osais.json .
COPY ./_temp/main_fastapi.py .
COPY ./_temp/main_flask.py .
COPY ./_temp/main_common.py .

COPY ./_temp/osais_auth.py .
COPY ./_temp/osais_config.py .
COPY ./_temp/osais_inference.py .
COPY ./_temp/osais_main.py .
COPY ./_temp/osais_pricing.py .
COPY ./_temp/osais_s3.py .
COPY ./_temp/osais_training.py .
COPY ./_temp/osais_utils.py .

# copy OSAIS -> AI
COPY ./_input/warmup.jpg ./_input/warmup.jpg
COPY ./gfpgan.json .

# overload config with those default settings
ENV ENGINE=gfpgan

# run as a server
CMD ["uvicorn", "main_fastapi:app", "--host", "0.0.0.0", "--port", "5012"]
