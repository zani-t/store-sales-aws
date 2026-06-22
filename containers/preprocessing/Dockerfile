FROM public.ecr.aws/lambda/python:3.12

COPY containers/preprocessing/app.py ${LAMBDA_TASK_ROOT}
COPY containers/preprocessing/requirements.txt ${LAMBDA_TASK_ROOT}
COPY tsf2_core ${LAMBDA_TASK_ROOT}/tsf2_core
RUN pip install -r requirements.txt --no-cache-dir

CMD ["app.lambda_handler"]
