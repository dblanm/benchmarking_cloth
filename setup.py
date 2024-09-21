from setuptools import setup

setup(
    name="bcm",
    packages=["bcm"],
    version="1.0",
    install_requires=[
        "imageio",
        "gym",
        "matplotlib",
        "mujoco",
        "sim_utils @ git+ssh://git@github.com/Barbany/sim_utils.git",
    ],
    extras_require={
        "formatters": ["black", "isort", "flake8"],
        "gpyopt": ["GPy @ git+ssh://git@github.com/Barbany/GPy.git", "GPyOpt"],
    },
)
