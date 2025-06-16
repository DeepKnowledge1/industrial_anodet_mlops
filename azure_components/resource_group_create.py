import argparse
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient

def create_resource_group(subscription_id, resource_group_name, location, tags):
    """
    Create an Azure Resource Group using the Azure SDK for Python, with optional tags.
    """
    credential = DefaultAzureCredential()
    resource_client = ResourceManagementClient(credential, subscription_id)
    resource_group_params = {
        "location": location,
        "tags": tags
    }
    print(f"Creating Resource Group '{resource_group_name}' in '{location}' with tags {tags}...")
    resource_group = resource_client.resource_groups.create_or_update(
        resource_group_name, resource_group_params
    )
    print(f"Resource Group '{resource_group.name}' created successfully!")
    print(f"Location: {resource_group.location}")
    print(f"Tags: {resource_group.tags}")

def main():
    parser = argparse.ArgumentParser(description="Create an Azure Resource Group with optional tags.")
    parser.add_argument("--subscription_id", type=str, default="5eab4ecc-5ecf-4754-802d-6da984293b70")
    parser.add_argument("--resource_group_name", type=str, default="rg-ml-industrial-dev")
    parser.add_argument("--location", type=str, default="eastus")
    parser.add_argument("--tags", nargs="*", help="Tags in key=value format (e.g. env=dev owner=deepK)")

    args = parser.parse_args()

    # Convert list of key=value pairs into dictionary
    tags_dict = {}
    if args.tags:
        for tag in args.tags:
            if "=" in tag:
                key, value = tag.split("=", 1)
                tags_dict[key] = value
            else:
                print(f"Ignoring malformed tag: '{tag}' (expected format key=value)")

    create_resource_group(args.subscription_id, args.resource_group_name, args.location, tags_dict)

if __name__ == "__main__":
    main()



#python azure_components/resource_group_create.py  --subscription_id "5eab4ecc-5ecf-4754-802d-6da984293b70"  --resource_group_name rg-ml-industrial-dev   --location eastus  --tags project=industrial environment=dev department=ml

# python azure_components/resource_group_create.py `
#   --subscription_id "5eab4ecc-5ecf-4754-802d-6da984293b70" `
#   --resource_group_name rg-ml-industrial-dev `
#   --location eastus `
#   --tags project=industrial environment=dev department=ml



# Pure Azure CLI v2 command to create a resource group
# az group create `
#   --subscription 5eab4ecc-5ecf-4754-802d-6da984293b70 `
#   --name rg-ml-industrial-dev `
#   --location eastus `
#   --tags project=industrial environment=dev department=ml

