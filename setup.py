import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='IceCharacterisation_XFEL',
    version='0.0.3',
    author='Michael Hassett',
    author_email='s3717891@student.rmit.edu.au',
    description='Characterisation of Ice from XFEL diffraction',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mph759/IceCharacterisation_XFEL',
    project_urls = {
        "Bug Tracker": "https://github.com/mph759/IceCharacterisation_XFEL/issues"
    },
    license='MIT',
    packages=['IceCharacterisation_XFEL'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'pathlib','h5py','pyFAI','mpl_toolkits','os','paltools','pandas','astropy','PIL'],
)