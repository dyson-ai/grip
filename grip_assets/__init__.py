import os


def get_data_path():
    resdir = os.path.join(os.path.dirname(__file__))
    return resdir


def fetch_data():
    os.system("rm -rf /tmp/grip_data > /dev/null 2>&1")
    os.system(
        "git clone --depth=1 ssh://git@stash.dyson.global.corp:7999/fjd/grip_data.git /tmp/grip_data"
    )
    os.system("pip3 install /tmp/grip_data")
