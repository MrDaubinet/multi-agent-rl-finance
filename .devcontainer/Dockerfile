# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.231.2/containers/python-3-anaconda/.devcontainer/base.Dockerfile
FROM rayproject/ray-ml

# create conda environment
# RUN conda env create --file requirements.txt

# # Make RUN commands use the new environment:
SHELL ["pip", "install", "tensortrade"]

# # SHELL ["conda", "create", "--name" <env_name> --file requirements.txt]

# # Demonstrate the environment is activated:
# RUN echo "Make sure tensortrade is installed:"
# RUN python -c "import tensortrade"
