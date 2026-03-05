# Helper file to build and run Docker images based on inferred module data.
import os
import shutil
import subprocess
from time import sleep

try:
    import docker as docker_sdk  # type: ignore
except Exception:
    docker_sdk = None


class DockerHelper:
    def __init__(self, logging=False, image_name="", dockerfile_name="", container_name="") -> None:
        self.dockerfile_out = ""
        self.image_name = image_name
        self.dockerfile_name = dockerfile_name
        self.container_name = container_name
        self.logging = logging
        self.previous_error = {"error_message": "", "module": ""}
        self.client = None
        self._use_sdk = False

        if docker_sdk is not None:
            try:
                self.client = docker_sdk.from_env()
                self._use_sdk = True
            except Exception as exc:
                if self.logging:
                    print(f"Docker SDK unavailable, falling back to Docker CLI: {exc}")

        if not self._use_sdk and shutil.which("docker") is None:
            raise RuntimeError(
                "Docker is not available. Install Docker CLI (and optionally docker Python SDK)."
            )

    def query_docker(self):
        if self._use_sdk and self.client is not None:
            return self.client.api.images()
        completed = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return []
        return [line.strip() for line in completed.stdout.splitlines() if line.strip()]

    # Breaks down the file path to get the folder and the file name
    # file: The path to the file
    def get_project_dir(self, file):
        split_path = file.split("/")
        file_path = "/".join(split_path[:-1])
        file_name = split_path[-1]
        dir_name = split_path[-2]
        return file_path, dir_name, file_name

    # Creates the dockerfile based on the llm information
    # llm_out: contains the python version and modules
    # file: The provided file with path
    def create_dockerfile(self, llm_out, file):
        project_dir, dir_name, project_file = self.get_project_dir(file)
        self.dockerfile_out = ""
        self.dockerfile_out += "# FROM is the found expected Python version\n"
        self.dockerfile_out += f"FROM python:{llm_out['python_version']}\n"
        self.dockerfile_out += "# Set the working directory to /app\n"
        self.dockerfile_out += "WORKDIR /app\n"

        self.dockerfile_out += "# Add install commands for all of the python modules\n"
        self.dockerfile_out += 'RUN ["pip","install","--upgrade","pip"]\n'
        python_modules = llm_out["python_modules"]
        if self.logging:
            print(python_modules)
        for module in python_modules:
            if isinstance(module, dict):
                name = module["module"]
                version = module["version"]
            else:
                name = module
                version = python_modules[module]

            if isinstance(version, str):
                self.dockerfile_out += (
                    'RUN ["pip","install","--trusted-host","pypi.python.org","--default-timeout=100",'
                    f'"{name}=={version}"]\n'
                )
            else:
                self.dockerfile_out += (
                    'RUN ["pip","install","--trusted-host","pypi.python.org","--default-timeout=100",'
                    f'"{name}=={version[0]}"]\n'
                )

        self.dockerfile_out += "# Copy the specified directory to /app\n"
        self.dockerfile_out += f"COPY {project_file} /app\n"
        self.dockerfile_out += "# Run the specified python file\n"
        self.dockerfile_out += f'CMD ["python", "/app/{project_file}"]'

        self.image_name = f"test/pllm:{dir_name}_{llm_out['python_version']}"
        self.container_name = f"{dir_name}_{llm_out['python_version']}"
        self.dockerfile_name = f"Dockerfile-llm-{llm_out['python_version']}"
        with open(f"{project_dir}/{self.dockerfile_name}", "w") as file_handle:
            file_handle.write(self.dockerfile_out)

    # Uses Docker SDK or CLI to build created dockerfiles.
    # Returns (success, error_output)
    def build_dockerfile(self, path, dockerfile=None):
        if not dockerfile:
            dockerfile = self.dockerfile_name
        project_dir, _dir_name, _project_file = self.get_project_dir(path)

        if self._use_sdk and self.client is not None:
            error_lines = ""
            for line in self.client.api.build(path=project_dir, dockerfile=dockerfile, forcerm=True, tag=self.image_name):
                decoded_line = line.decode("utf-8")
                if "ERROR" in decoded_line or "Could not fetch URL" in decoded_line or "errorDetail" in decoded_line:
                    error_lines += decoded_line
                if self.logging:
                    print(decoded_line)
            if error_lines == "":
                return True, ""
            return False, error_lines

        command = ["docker", "build", "-f", dockerfile, "-t", self.image_name, project_dir]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        output = f"{completed.stdout or ''}\n{completed.stderr or ''}".strip()
        if self.logging and output:
            print(output)
        if completed.returncode == 0:
            return True, ""
        return False, output

    def delete_container(self):
        if self._use_sdk and self.client is not None:
            try:
                self.client.containers.get(self.container_name).remove(v=True, force=True)
            except Exception as exc:
                if self.logging:
                    print(exc)
            return
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
            text=True,
            check=False,
        )

    def delete_image(self):
        if self._use_sdk and self.client is not None:
            try:
                self.client.images.remove(image=self.image_name, force=True)
            except Exception as exc:
                if self.logging:
                    print(exc)
            return
        subprocess.run(
            ["docker", "image", "rm", "-f", self.image_name],
            capture_output=True,
            text=True,
            check=False,
        )

    # Runs the container we built to see if the python snippet runs.
    # Returns logs for analysis.
    def run_container_test(self):
        self.delete_container()

        if self._use_sdk and self.client is not None:
            logs = ""
            container = None
            try:
                container = self.client.containers.create(self.image_name, name=self.container_name)
                container.start()
                sleep(10)
                while container.status == "running":
                    sleep(5)
                if self.logging:
                    print(container.status)
                logs = container.logs()
                container.remove(v=True, force=True)
                container = None
            except Exception as exc:
                if self.logging:
                    print(exc)
                if container:
                    while container.status == "running":
                        sleep(5)
                    if self.logging:
                        print(container.status)
                    logs = container.logs()
                    container.remove(v=True, force=True)
                    container = None

            return logs.decode("utf-8") if isinstance(logs, (bytes, bytearray)) else str(logs)

        command = ["docker", "run", "--name", self.container_name, self.image_name]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        output = f"{completed.stdout or ''}\n{completed.stderr or ''}".strip()
        self.delete_container()
        return output


def main():
    dh = DockerHelper(logging=True, image_name="woof:meow", dockerfile_name="", container_name="")
    dh.run_container_test()


if __name__ == "__main__":
    main()
    print("Done")
