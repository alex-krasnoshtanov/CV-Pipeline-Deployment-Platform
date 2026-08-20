FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest

COPY conda.yml /tmp/conda.yml
COPY cv_pipeline-0.1.0-py3-none-any.whl /tmp/cv_pipeline-0.1.0-py3-none-any.whl

RUN conda env update -n base -f /tmp/conda.yml && \
    pip install --no-deps /tmp/cv_pipeline-0.1.0-py3-none-any.whl && \
    conda clean -ay && \
    rm /tmp/conda.yml /tmp/cv_pipeline-0.1.0-py3-none-any.whl
