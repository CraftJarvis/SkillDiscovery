# Create the directory structure if it doesn't exist
mkdir -p weights
mkdir -p models
mkdir -p data

# Foundation VPT model
wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-3x.model -P models
wget https://openaipublic.blob.core.windows.net/minecraft-rl/models/bc-early-game-3x.weights -P weights

# IDM model
wget https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model -P models
wget https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights -P weights

# data index file
wget https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/all_7xx_Apr_6.json -P data