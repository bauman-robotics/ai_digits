Jupyter
https://www.altlinux.org/Jupyter_Notebook
apt-get install spyder
apt-get install python3-module-notebook

$ su -
# apt-get update
# apt-get install python3-module-notebook -- err
# apt-get install python3-module-nbclient
# logout

jupyter notebook

Сделаем данное окружение доступным в качестве ядра для Jupyter Notebook:

(venv) $ python -m ipykernel install --user --name=venv

Выйдем из окружения и запустим Jupyter Notebook

(venv) $ deactivate
$ jupyter notebook

//==============
sudo apt-get update
sudo apt-get install spyder
sudo apt install epiphany jupyter-notebook  # Ubuntu 22.04 and later
jupyter notebook --generate-config
nano /home/andrey/.jupyter/jupyter_notebook_config.py
	c.NotebookApp.browser = '/usr/bin/epiphany'
	c.NotebookApp.browser = '/usr/bin/firefox'

Spyder is an open-source cross-platform integrated development environment (IDE) 
for scientific programming in the Python language. Spyder integrates with a number of prominent 
	packages in the scientific Python stack, including NumPy, SciPy, Matplotlib, pandas, IPython, SymPy and Cython, 
as well as other open-source software.[4][5] It is released under the MIT license.[6] 

epiphany
Create jupyter_notebook_config.py by:

jupyter notebook --generate-config # type y for yes at the prompt

Then open ~/.jupyter/jupyter_notebook_config.py for editing in a text editor and change:
nano ~/.jupyter/jupyter_notebook_config.py

# c.NotebookApp.browser = ''

to:

c.NotebookApp.browser = '/usr/bin/epiphany'

andrey@Lenovo:~/projects/ai$ jupyter notebook --generate-config
Writing default config to: /home/andrey/.jupyter/jupyter_notebook_config.py

which firefox
/usr/bin/firefox

https://askubuntu.com/questions/737094/jupyter-notebook-installation

~/.local/share

jupyter notebook
http://localhost:8888/?token=c5165c01e976455eb89ac6eb2fcf4b2a3f8f87f67714cea0


//======================

jupyter notebook  
new (in browser) Python3 kernel 


//=====================
https://vc.ru/newtechaudit/416183-python-venv-i-jupyter-ipykernel-dlya-samyh-zelenyh

1. venv

2. venv activate 

pip install notebook

sudo python3 -m ipykernel install --name=env_kernel

jupyter notebook

В случае если нужно удалить kernel можно воспользоваться командой jupyter kernelspec remove *название kernel.

Для того, чтобы деактивировать наше виртуальное окружение используем команду deactivate