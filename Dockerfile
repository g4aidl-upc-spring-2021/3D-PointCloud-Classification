# set base image (host OS)
FROM python:3.7-slim-buster

# set the working directory in the container
WORKDIR /src

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install -r requirements.txt

# Copy code to the working directory
COPY src/ .

# command to run on container start
ENTRYPOINT ["python", "main.py"]