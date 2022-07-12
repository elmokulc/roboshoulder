import os
import pwd
from string import Template
import json
import shutil


username = os.environ.get("USER")
uid = str(pwd.getpwnam(os.environ.get("USER")).pw_uid)
gid = str(pwd.getpwnam(os.environ.get("USER")).pw_gid)

dockerfile = "Dockerfile"
tag = "0.0.py37"
conda_env_name = "roboshoulder"
image_name = f"celmo/roboshoulder:{tag}"

data = {
    "USERNAME": username,
    "UID": uid,
    "GID": gid,
    "TAG": tag,
    "CONDA_ENV_NAME": conda_env_name,
    "IMAGE_NAME": image_name,
    "DOCKERFILE": dockerfile,
}
dct = Template(open("docker-compose_template.yml").read().strip()).substitute(data)


# MY VOLUMES
if not os.path.isfile("my_volumes.json"):
    shutil.copy("my_volumes_template.json", "my_volumes.json")
my_volumes = json.load(open("my_volumes.json"))
if len(my_volumes) != 0:
    dct += "\n" + 4 * " " + "volumes:" + "\n" + 6 * " "
for volume in my_volumes:
    dct += "- {0}:{1}".format(volume["host"], volume["target"])
    if "read_only" in volume.keys() and volume["read_only"]:
        dct += ":ro"
    dct += "\n" + 6 * " "

dct = dct.strip()


# if len(add_to_pythonpath) == 0:
#     PYTHONPATH = ""
# else:
#     PYTHONPATH = "\n".join([f"PYTHONPATH=$PYTHONPATH:{target}" for target in add_to_pythonpath])
open("docker-compose.yml", "w").write(dct)
# open("Dockerfile").write(dft.substitute(PYTHONPATH=PYTHONPATH))
