from setuptools import setup, find_packages

# Function to read the requirements from the requirements.txt file
def read_requirements(file):
    try:
        with open(file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Warning: {file} not found.")
        return []

# Function to read the long description from the README.md file
def read_long_description(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {file} not found.")
        return "This is my first ML project."

setup(
    name='ML_PROJECT2',
    version='0.1.0',
    description='This is my first ML project',
    long_description=read_long_description('README.md'),
    long_description_content_type='text/markdown',
    author='Abhaya',
    author_email='aby.acharya.aws1@gmail.com',
    # url='https://github.com/yourusername/my_project',
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt'),
)
