from setuptools import setup, find_packages

setup(
    name="signature-project",
    version="0.0.1",
    author="Rasheed Ibraheem",
    author_email="R.O.Ibraheem@sms.ed.ac.uk",
    maintainer="Rasheed Ibraheem",
    maintainer_email="R.O.Ibraheem@sms.ed.ac.uk",
    description="Early prediction of Remaining Useful Life for Lithium-ion cells using only signatures of voltage curves at 4 minute sampling rates",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Rasheed19/signature-project.git",
    project_urls={
        "Bug Tracker": "https://github.com/Rasheed19/signature-project.git/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        'mlflow==2.4.0',
        'iisignature==0.24',
        'pandas==2.0.2',
        'numpy==1.25.2',
        'matplotlib==3.7.1',
        'scipy==1.10.1',
        'scikit-learn==1.2.2',
        'rrct==1.0.4',
        'DateTime==5.1',
        'h5py==3.8.0',
        'xgboost==1.7.5',
        'seaborn==0.12.2',
        'pingouin==0.5.3'
    ]
)
