from setuptools import setup, find_packages

setup(
    name='API_MLE',  # Cambia esto por el nombre real de tu paquete
    version='0.1',  # Ajusta esto según la versión de tu paquete
    packages=find_packages(),  # Esto incluirá todos los subpaquetes automáticamente
    scripts=['scripts/api.py'],  # Ajusta la ruta si tu script está en otro lugar
    install_requires=[
        'pandas',
        'numpy',
        'ydata-profiling',  # Asegúrate de que el nombre del paquete esté correcto
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'pickle-mixin',  # Usualmente se usa solo 'pickle' que ya está en la biblioteca estándar de Python
        'flask',
        'requests'
    ],
    entry_points={
        'console_scripts': [
            'nombre_del_comando = API_MLE.scripts.api:main',
        ],
    },
)
