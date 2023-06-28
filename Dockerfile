##
##      To build the AI_GFPGAN docker image
##

# base stuff
FROM yeepeekoo/public:ai_gfpgan_

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
