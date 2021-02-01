FROM python:3.8.6

COPY . /piv/

WORKDIR /piv

RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD ["python", "rigidtransforms.py", "rgbimgs.txt" , "depthimgs.txt" , "cameracalib.txt" , "transforms.txt"]