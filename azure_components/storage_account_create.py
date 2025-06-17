import argparse
from azure.identity import DefaultAzureCredential
from azure.mgmt.storage import StorageManagementClient
from azure.storage.blob import BlobServiceClient

def create_storage_account(subscription_id, resource_group, storage_account_name, region):
    credential = DefaultAzureCredential()
    storage_client = StorageManagementClient(credential, subscription_id)

    print(f"Creating Storage Account '{storage_account_name}' in '{region}'...")
    storage_account_params = {
        "sku": {"name": "Standard_LRS"},
        "kind": "BlobStorage",
        "location": region,
        "tags": {"environment": "dev", "project": "industrial"}
    }

    poller = storage_client.storage_accounts.begin_create(
        resource_group_name=resource_group,
        account_name=storage_account_name,
        parameters=storage_account_params
    )
    account = poller.result()
    print(f"Storage Account '{account.name}' created successfully!")

def connect_to_blob_service(storage_account_name):
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net/", 
        credential=credential
    )
    print("Available containers:")
    for container in blob_service_client.list_containers():
        print(f" - {container['name']}")

def main():
    parser = argparse.ArgumentParser(description="Create Azure Storage Account")
    parser.add_argument("--subscription_id", type=str, default="your-sub-id")
    parser.add_argument("--resource_group", type=str, default="rg-ml-industrial-dev")
    parser.add_argument("--storage_account_name", type=str, default="industrialdevstorage")
    parser.add_argument("--region", type=str, default="eastus")
    args = parser.parse_args()

    create_storage_account(
        args.subscription_id,
        args.resource_group,
        args.storage_account_name,
        args.region
    )
    connect_to_blob_service(args.storage_account_name)

if __name__ == "__main__":
    main()