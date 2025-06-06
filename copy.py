import os
import shutil

def copy_py_to_txt(source_dirs, dest_dir='./txt/'):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created destination directory: {dest_dir}")

    copied_files = 0
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory does not exist: {source_dir}")
            continue
            
        print(f"Scanning directory: {source_dir}")
        
        for filename in os.listdir(source_dir):
            if filename.endswith('.py') and filename != '__init__.py':
                source_file = os.path.join(source_dir, filename)
                # Change extension from .py to .txt
                dest_filename = filename[:-3] + '.txt'
                dest_file = os.path.join(dest_dir, dest_filename)
                
                try:
                    shutil.copy2(source_file, dest_file)
                    print(f"Copied: {source_file} -> {dest_file}")
                    copied_files += 1
                except Exception as e:
                    print(f"Error copying {source_file}: {e}")
    
    print(f"\nTotal files copied: {copied_files}")

if __name__ == "__main__":
    # Easy configuration - just modify this list with your directories
    source_directories = [
        './agents',
        './core',
        './config',
        './tools',
        # Add more directories here as needed
    ]
    
    # Run the copy operation
    copy_py_to_txt(source_directories)
