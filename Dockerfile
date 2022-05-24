FROM python:3.8-slim-buster
WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt -i  https://mirrors.aliyun.com/pypi/simple/
CMD ["python", "web-onnx.py"]
EXPOSE 5001