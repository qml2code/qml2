import os
import subprocess

workdir = os.getcwd()

doc_dir = os.path.dirname(__file__)

os.chdir(doc_dir)

try:
    completed_doxygen = subprocess.run("doxygen")
    assert completed_doxygen.returncode == 0, "doxygen failure"
    main_html = "html/index.html"
    assert os.path.isfile(main_html)
    with open("../manual.html", "w") as f:
        print('<meta http-equiv="REFRESH" content="0;URL=./doxygen/' + main_html + '">', file=f)
except FileNotFoundError:
    print("doxygen not found; install it by running:\nconda install conda-forge::doxygen")

os.chdir(workdir)
