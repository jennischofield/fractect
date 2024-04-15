# fractect
A few things before you try to run the project - I don't know the status of your machine, so we're going to download all the necessary bits and bobs beforehand. We'll be downloading things into a virtual environment (venv) so you can delete it easily after testing!

1) Open up a terminal (Terminal for both Windows and Mac. If you're using Linux, then you can convert your own commands, you have to have command line experience for that)
2) Make sure you have python3 downloaded, if this doesn't work, go download it online, then verify: python --version
3) Create a virtual environment: python -m venv tester-pack
4) Confirm that the virtual environment was created successfully and activate it: 
	Windows: tester-pack\Scripts\activate    MacOS/Unix: source tester-pack/bin/activate
5) Now we have our venv, we need to download all the necessary packages for this to work. We'll be using pip to do so, but if you're on it, you can use conda, or whatever package manager you'd like, just adjust the commands as you'd like. Verify that pip is installed by:
	Windows: py -m ensurepip --upgrade       MacOS/Unix: python -m ensurepip --upgrade

6) Install dicom2jpg: pip install dicom2jpg

7) Install flask: pip install flask

8) Install albumetations: pip install albumentations

9) Install torchmetrics: pip install torchmetrics

10) Install tqdm: pip install tqdm

11) Install torchvision: pip install torchvision

12) Install matplotlib: pip install matplotlib

13) This should be all the requirements now. Navigate to the folder containing the code. Example: To get to Downloads, do cd Downloads. Repeat until you're at the tester-package folder.

14) Try spinning up the server by running: python -m flask --app fractect run
