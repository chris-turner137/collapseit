# Export environment variables for this repository.
# Usage: source ~/Programming/collapseit/export.sh
# Note: The script must be passed it's fully qualified path.
export PYTHONPATH="$(pwd)/packages:$(pwd)/submodules/packages:$PYTHONPATH"
SOURCE=$(dirname "${BASH_SOURCE[0]}")
export PYTHONPATH="$SOURCE/packages:$PYTHONPATH"
