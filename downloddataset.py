import shutil
from pathlib import Path
import kagglehub

# Download to cache
cache_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

# Destination folder in your project
dest_path = Path("data")
dest_path.mkdir(exist_ok=True)

# Copy files from cache to your project's data folder
for item in Path(cache_path).iterdir():
    if item.is_file():
        shutil.copy2(item, dest_path / item.name)

print(f"Dataset successfully copied to: {dest_path.resolve()}")
