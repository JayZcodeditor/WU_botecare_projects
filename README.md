# WU_botecare_projects

**Part for Local Computer**

1. First, install any required libraries using pip.

2. Run the following command to start the app:

### Command
```bash
uvicorn app:app --reload
```



**Part for Docker**
1. Install Docker Desktop from the following link: Docker Desktop Installer.

2. Navigate to the directory containing your WU_botecare project.

3. nRun the following command to build the Docker image:

### Command
```bash
docker build -t <image_name>:<tag> <path_to_Dockerfile>
```


➡️ Replace <image_name> with the desired name for your Docker image, <tag> with an optional tag, and <path_to_Dockerfile> with the path to your Dockerfile.
