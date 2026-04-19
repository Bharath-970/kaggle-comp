#!/bin/bash
# 1. Prepare clean dataset folder
rm -rf kaggle_dataset
mkdir -p kaggle_dataset
zip -r code.zip src scripts
mv code.zip kaggle_dataset/

# 2. Create the metadata again to be 100% sure
cat << 'EOF' > kaggle_dataset/dataset-metadata.json
{
  "title": "neurogolf-source",
  "id": "bharath111l/neurogolf-source",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# 3. Create or Update the dataset
# We try 'create' first; if it exists, we run 'version'
./.venv/bin/kaggle datasets create -p kaggle_dataset || ./.venv/bin/kaggle datasets version -p kaggle_dataset -m "Force Update"

# 4. Push the kernel
./.venv/bin/kaggle kernels push -p scripts/kaggle_deployment
