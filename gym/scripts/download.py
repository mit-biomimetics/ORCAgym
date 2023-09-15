import wandb


# download the specified WandB artifact to the artifacts parent directory
def download_artifact_code(artifact_name):
    run = wandb.init()
    artifact = \
        run.use_artifact(artifact_name, type='code')
    artifact.download('../../artifacts/'+artifact_name)


# the name WandB name of the artifact to download
artifact_name = 'ajm4/logging_test/source-367gxqhp:v0'
download_artifact_code(artifact_name)
