import argparse
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace


def create_workspace(subscription_id, resource_group, workspace_name, region):
    """
    Create an Azure ML Workspace using the Azure SDK for Python.
    """
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group)

    print(f"Creating Azure ML Workspace '{workspace_name}' in resource group '{resource_group}'...")
    
    workspace = Workspace(
        name=workspace_name,
        location=region,
        display_name="Industrial ML Workspace",
        description="Workspace for industrial ML projects",
        tags={"environment": "development", "project": "mlops"},
    )

    ml_client.workspaces.begin_create(workspace)
    print(f"Workspace '{workspace_name}' created successfully!")


def main():
    parser = argparse.ArgumentParser(description="Create an Azure ML Workspace via Python SDK v2.")
    parser.add_argument("--subscription_id", type=str, required=True, help="Azure Subscription ID")
    parser.add_argument("--resource_group", type=str, required=True, help="Resource group name")
    parser.add_argument("--workspace_name", type=str, required=True, help="Workspace name")
    parser.add_argument("--region", type=str, required=True, help="Azure region for the workspace (e.g., eastus)")
    
    args = parser.parse_args()
    create_workspace(args.subscription_id, args.resource_group, args.workspace_name, args.region)


if __name__ == "__main__":
    main()
    
    
    