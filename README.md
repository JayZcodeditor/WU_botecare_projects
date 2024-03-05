WU_botecare_projects
Part for Local Computer
First, install any required libraries using pip.

Run the following command to start the app:

bash
Copy code
uvicorn app:app --reload
Part for Docker
Install Docker Desktop from the following link: Docker Desktop Installer.

Navigate to the directory containing your WU_botecare project.

Run the following command to build the Docker image:

bash
Copy code
docker build -t <image_name>:<tag> <path_to_Dockerfile>
Replace <image_name> with the desired name for your Docker image, <tag> with an optional tag, and <path_to_Dockerfile> with the path to your Dockerfile.
