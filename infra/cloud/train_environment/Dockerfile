FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest

# The wheel is a build artefact and is not tracked. Build it before building
# this image:
#
#   uv build --wheel packages/cv-pipeline -o <this directory>
#
# CD does this for you in deploy-endpoint.yml. env_registration.py and
# submit_training.py check for it and say the same thing if it is missing.
COPY conda.yml /tmp/conda.yml
COPY cv_pipeline-0.1.0-py3-none-any.whl /tmp/cv_pipeline-0.1.0-py3-none-any.whl

RUN conda env update -n base -f /tmp/conda.yml && \
    pip install --no-deps /tmp/cv_pipeline-0.1.0-py3-none-any.whl && \
    conda clean -ay && \
    rm /tmp/conda.yml /tmp/cv_pipeline-0.1.0-py3-none-any.whl
