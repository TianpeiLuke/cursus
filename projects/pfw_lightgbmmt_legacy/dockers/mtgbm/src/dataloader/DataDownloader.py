from secure_ai_sandbox_python_lib.session import Session


class DataDownloader:
    """
    The tool to download the data
    You will need to have a workflow id to pull the data

    Parameter
    ----------
    region: The region where the workflow exist

    path: The path of the folder where the data will be download to

    workflow_id: The workflow ID from DAWS

    """

    def __init__(
        self,
        region="eu-west-1",
        resource="DAWSWorkflowFilesDownloader",
        path="/home/ec2-user/SageMaker/data",
        workflow_id="tayefan.train.BFS.ConsortiumXGBoostSuspectQueueModel.2022-06-30-18657-quiet-mice",
    ):
        self.region = region
        self.resource = resource
        self.path = path
        self.workflow_id = workflow_id

    def download_data(self):
        session = Session(".")
        daws_downloader = session.resource(self.resource)
        daws_downloader.download_workflow_files(
            aws_region=self.region,
            workflow_id=self.workflow_id,
            output_path=self.path,
            input_directory="data-collection/TESTING",
        )

        daws_downloader.download_workflow_files(
            aws_region=self.region,
            workflow_id=self.workflow_id,
            output_path=self.path,
            input_directory="data-collection/TRAINING",
        )
