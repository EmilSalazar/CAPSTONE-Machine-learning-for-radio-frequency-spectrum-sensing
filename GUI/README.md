# GUI

This directory contains our GUI implementation. We utilized Flask to create a locally hosted server that opens a webpage displaying a video of a spectrogram. At the top of the page, the model predictions for either Bluetooth or WiFi are shown, updating every second.

The app.py file defines two routes: one for rendering index.html from our templates folder, and another for updating and displaying the prediction of either WiFi or Bluetooth. Our index.html uses basic HTML and CSS, along with a small amount of JavaScript. Any images and videos used on the webpage are stored in the static folder.

If the trained_model.h5 file is deleted, it can be restored by running the train.py file in the training directory and copying the model back into this folder. Additionally, the code can be modified so that instead of saving the model in the testing directory, it is saved directly in this one.

To launch the GUI, run the app.py file. This will start the locally hosted server and automatically open the webpage.

> [!NOTE]
> Make sure you have flask installed!!




