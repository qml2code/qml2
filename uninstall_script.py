import os
import shutil
import subprocess
import sys

os.chdir("/")
p = subprocess.run(["python", "-c", "import qml2; print(qml2.__file__)"], capture_output=True)
if p.returncode == 1:
    quit()
qml2_init_file = p.stdout.decode().strip()
qml2_folder = os.path.dirname(qml2_init_file)

subprocess.run([sys.argv[1], "uninstall", "--yes", "qml2"])

if os.path.isdir(qml2_folder):
    # probably contains some Numba cache
    shutil.rmtree(qml2_folder)
