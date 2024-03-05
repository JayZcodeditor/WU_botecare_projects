WU_botecare_projects
Part for Local Computer
1.First, install any required libraries using pip.
2.Run the following command to start the app:

Markdown

uvicorn app:app --reload
```python

Part for Docker
Install Docker Desktop from the following link: Docker Desktop Installer.

Navigate to the directory containing your WU_botecare project.

Run the following command to build the Docker image:

Markdown

docker build -t <image_name>:<tag> <path_to_Dockerfile>
```python

➡️ Replace <image_name> with the desired name for your Docker image, <tag> with an optional tag, and <path_to_Dockerfile> with the path to your Dockerfile.
