import zipfile
import io
import requests
import os
import time
import datetime

from elliot.run import run_experiment

print("Done! We are now starting the Elliot's experiment")
# run_experiment("config_files/ciao_basic_configuration_v030.yml")
run_experiment("config_files/social_recomm_configuration_toy_dataset.yml")

# run_experiment("config_files/social_recomm_configuration_ciao.yml")
# run_experiment("config_files/social_recomm_configuration_epinions.yml")
