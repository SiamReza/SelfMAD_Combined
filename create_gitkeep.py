import os

def create_gitkeep_files(base_dir):
    """Create .gitkeep files in all subdirectories of base_dir"""
    for root, dirs, files in os.walk(base_dir):
        # Skip .git directories
        if '.git' in root:
            continue
            
        # Create .gitkeep file in each directory
        gitkeep_path = os.path.join(root, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, 'w') as f:
                pass  # Create empty file
            print(f"Created {gitkeep_path}")

if __name__ == "__main__":
    # Create .gitkeep files in datasets directories
    create_gitkeep_files("datasets")
    create_gitkeep_files("SelfMAD-siam/datasets")
    
    print("Done creating .gitkeep files to preserve directory structure.")
