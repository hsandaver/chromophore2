# Use a Miniconda base image
FROM continuumio/miniconda3

# Install mamba (a faster, lighter-weight installer)
RUN conda install -n base -c conda-forge mamba -y

# Use mamba to install RDKit and your other dependencies
RUN mamba install -n base -c conda-forge rdkit streamlit pandas altair pillow py3Dmol -y

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Expose the port Streamlit uses (default is 8501)
EXPOSE 8501

# Run the Streamlit app. The --server.enableCORS false flag is useful for Spaces.
CMD ["streamlit", "run", "app.py", "--server.enableCORS", "false"]