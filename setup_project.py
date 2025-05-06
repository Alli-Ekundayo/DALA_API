from pathlib import Path
import shutil

def create_project_structure():
    # Define the base directory
    base_dir = Path('itsekiri-translator')
    
    # Create main project directories
    dirs = [
        'app',
        'app/model',
        'data',
        'models',
        'tests'
    ]
    
    # Create directories
    for dir_path in dirs:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create empty Python files
    py_files = [
        'app/__init__.py',
        'app/api.py',
        'app/config.py',
        'app/preprocessing.py',
        'app/model/__init__.py',
        'app/model/translator.py',
        'app/model/tokenizers.py',
        'tests/__init__.py',
        'tests/test_api.py',
        'tests/test_model.py',
        'run.py',
        'train.py'
    ]
    
    for file_path in py_files:
        (base_dir / file_path).touch()
    
    # Create .gitkeep in models directory
    (base_dir / 'models/.gitkeep').touch()
    
    # Create Docker-related files
    docker_files = ['.dockerignore', 'Dockerfile']
    for file_path in docker_files:
        (base_dir / file_path).touch()
    
    # Create .gitignore with basic Python patterns
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
models/*
!models/.gitkeep
"""
    
    with open(base_dir / '.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    # Move the clean_itsekiri.csv to data directory if it exists
    source_csv = Path('data/clean_itsekiri.csv')
    if source_csv.exists():
        shutil.copy2(source_csv, base_dir / 'data/clean_itsekiri.csv')
    
    print("Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()