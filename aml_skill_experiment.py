'''
This experiment runs the model training in AML cluster compute
using AML Experiment and register the models in AML.
You can use this Experiment to re-run the model training
in AML.
'''

from azureml.core.runconfig import RunConfiguration, DEFAULT_CPU_IMAGE
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core import Experiment, Workspace, ScriptRunConfig
import os
from azureml.core import Model

# Download config.json file from AML workspace and keep it in root dir or you can mention the location inside from_config.
ws = Workspace.from_config()

cpu_cluster_name = 'your-cluster-compute'

# Create new runconfig object. Create deployment env
run_amlcompute = RunConfiguration()
run_amlcompute.target = cpu_cluster_name
run_amlcompute.environment.docker.enabled = True
run_amlcompute.environment.docker.base_image = DEFAULT_CPU_IMAGE
run_amlcompute.environment.python.user_managed_dependencies = False

# Specify CondaDependencies obj, add necessary packages
run_amlcompute.environment.python.conda_dependencies = CondaDependencies.create(
    conda_packages=['pandas', 'scikit-learn'],
    pip_packages=['azureml-sdk', 'tensorflow', 'keras', 'configparser', 'azureml-dataprep[pandas]'],
    pin_sdk_version=False
)

experiment_name = 'skill-classifier-training'
exp = Experiment(workspace=ws, name=experiment_name)

# Log matrices in AML workspace
logging = exp.start_logging()

script_folder = os.getcwd()
src = ScriptRunConfig(
    source_directory= '.',
    script= 'aml_skill_training.py',
    run_config = run_amlcompute
)

run = exp.submit(src)
run.wait_for_completion(show_output=True)

# Register Tokenizer and Keras classifier model in AML
run.register_model(
    model_name='skill_tokenizer.pkl',
    model_path='outputs/skill_tokenizer.pkl',
    description='Skill CLassifier Vectorizer',
    model_framework= Model.Framework.TFKERAS,
    model_framework_version='1.0.0'
)

run.register_model(
    model_name='skill_classifier',
    model_path='outputs/skill_classifier',
    description='Skill CLassifier Model',
    model_framework= Model.Framework.TFKERAS,
    model_framework_version='1.0.0'
)

logging.complete()




