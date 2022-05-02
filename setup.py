from setuptools import setup

package_name = 'f1tenth_planning'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Billy Zheng',
    maintainer_email='billyzheng.bz@gmail.com',
    description='F1TENTH Motion Planning Library',
    license='MIT License',
    tests_require=['pytest'],
)
