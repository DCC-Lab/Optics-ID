# **How to Install the Optics-Id App on any computer**

------

This tutorial should help you to install correctly the Optics-Id app, but more specificaly the microRamanView window. The first window, or filterView window, shall work on its own when the aforementioned installation is done.

## What is the Optics-Id app?

------

The Optics-Id app is your new best friend! It is designed to control simultaneously a stage, spectrometer and a light source of any kind. Its main purpose is to scan a sample of your choice and render an hyperspectral image of said sample. The user interface is quite intuitive, and will give you access to saved spectra, matrices, background information, a preview of the data, and much more.

## How to use the UI?

------

Once the program is rightfully installed, you can access [HOWTO-OpticsId](https://github.com/PyMarc2/Optics-ID) for more information on fonctionnalities and special features of the app.

## How to install Optics-Id on your computer?

------

1. First, you need to make sure that everything is up-to-date on your computer. In this version of the program, we are using a python interpreter 3.9 or 3.8, pycharm version 2021.1.3 (community edition), and BigSur (version 11.4) on macOS (click on the apple in the top-left corner of your screen, and then "about this Mac") or Windows 10 on a PC;

2. Once you have acces to the code and you have cloned your repository on your computer, make sure to create and install a virtual environment with the requirements given in the repository (if needed, here is the [HOWTO-venv](https://github.com/DCC-Lab/Documentation/blob/master/HOWTO/HOWTO-PythonVirtualEnvironment(venv).md) made by DCC-Lab);

3. Activate the venv;

4. One essential python module, installed in your virtual environment by the command presented hereunder, is pyusb.

   ```bash
   $ pip install -r requirements.txt
   ```

   However, since you probably created a brand new venv for the project, you will need to get the usb library as well, which can be tricky. Perhaps, the libusb is already downloaded on your computer, in which case you won't need to follow the next steps - directly skip to the step 5 - in order for your app to work perfectly. Otherwise, here are a few additional steps for you to read carefully. You will also need to install git on your computer (see steps 3.1-3.2).

   3.1* - Before downloading libusb, you will need [Homebrew](https://brew.sh/), a package manager complementary to pip. To install, simply write in your terminal (for macOS or Linux only):

   ```bash
   $ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   Please note that steps 3.1 and 3.2 are exectuted while the venv is activated.

   3.2** - Then, install git.

   ```bash
   $ brew install git
   ```

   3.3 - You can now deactivate your virtual environment simply by writing "deactivate" directly in your terminal, no matter where you may have ended up in your directories. 

   3.4 - Now that you are back in your local directories, install libusb with the following line of code:

   ```bash
   $ brew install libusb
   ```

   3.5 - Link libusb to the abovementionned module.

   ```bash
   $ brew link libusb
   ```

   This step can be asked for "by the terminal" itself. Thus, I would invite you to read carefully every error or warning message that could occur when downloading anything in Terminal. It is also possible that everything went well, therfore rendering this step futile. An additional step I used afterwards was:

   ```bash
   $ brew link --overwrite libusb
   ```

5. Now that Homebrew, libusb and every module cited in requirements.txt has been acquired, your work is done. All you have to do now is activate the virtual environment, run the opt-id file and admire your beautiful results!

You can now proceed with your research and let our app marvel you, as well as awaken your curiosity.

![raman8](/Users/justinemajor/Documents/Captures/raman8.png)

------

\* The following steps are designed to work on macOS specifically. The principal difference should lie in the protocol to download libusb.

** Even if you already have git installed on your computer, you will need it in your homebrew in order for libusb to work and connect with the pyusb module.