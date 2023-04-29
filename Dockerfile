FROM python:3.8-buster
RUN apt update && apt install -y portaudio19-dev python3-pyaudio libsndfile1-dev

WORKDIR /app
ADD requirements.txt .
RUN pip install -r requirements.txt

ADD model.h5 .
ADD recognize.py .

ENTRYPOINT ["python", "recognize.py", "--file"]
