FROM public.ecr.aws/lambda/python:3.12

COPY app.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt --no-cache-dir

CMD ["app.lambda_handler"]
