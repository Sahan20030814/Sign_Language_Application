import subprocess
import sys
import os

def get_base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def main():
    base_dir = get_base_dir()
    python_dir = os.path.join(base_dir, "python_embed")
    python_exe = os.path.join(python_dir, "pythonw.exe")
    app_script = os.path.join(base_dir, "app.py")

    env = os.environ.copy()
    env["TCL_LIBRARY"] = os.path.join(python_dir, "tcl", "tcl8.6")
    env["TK_LIBRARY"] = os.path.join(python_dir, "tcl", "tk8.6")

    subprocess.Popen([python_exe, app_script], cwd=base_dir, env=env)

if __name__ == "__main__":
    main()