FROM capless/capless-docker:jupyter
RUN pip install --upgrade pip poetry
COPY . /code
RUN poetry run pip install --upgrade pip
RUN poetry install
