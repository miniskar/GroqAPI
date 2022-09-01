import subprocess
import re
import shutil
import os
from groq.common import print_utils


def find_sdk(package: str):
    sdk_available = False
    INTERNAL = os.getenv("INTERNAL")
    # Check if SDK is not being used
    if INTERNAL == "True":
        sdk_available = True
        return sdk_available

    cmd = ["apt-cache", "policy", package]
    apt_cache_policy = subprocess.check_output(cmd).decode("utf-8").split("\n")
    if len(apt_cache_policy) == 1:
        print_utils.err(
            f"No {package} found. Visit https://support.groq.com to download and install the GroqWare Suite. If you have already installed GroqWare, ensure that you've added /opt/groq/runtime/site-packages to your PYTHONPATH."
        )
        return
    else:
        sdk_version = re.search(r"(?<=Installed: ).*", apt_cache_policy[1]).group()
        if sdk_version != "None":
            return True


def get_num_chips_available(pci_devices=None):

    INTERNAL = os.getenv("INTERNAL")
    # Check if SDK is not being used
    if INTERNAL == "True":
        chips_available = "True"
        return chips_available

    # Check if we have access to lspci
    if not shutil.which("lspci"):
        print("No access to lspci")
        return

    # Capture the list of pci devices on the system using the linux lspci utility
    if pci_devices is None:
        pci_devices = (
            subprocess.check_output(["/usr/bin/lspci", "-n"])
            .decode("utf-8")
            .split("\n")
        )

    # Unique registered vendor id: 1de0, and device id: "0000"
    groq_card_id = "1de0:0000"

    # number of chips per device: "1de0:0000":1
    chips_per_card = 1

    # Sum the number of GroqCards in the list of devices
    num_cards = 0
    for device in pci_devices:
        if groq_card_id in device:
            num_cards += 1

    # Calculate total number of chips
    num_chips_available = num_cards * chips_per_card

    return num_chips_available
