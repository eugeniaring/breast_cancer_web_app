# Web Application to detect Breast Cancer in Ultrasound images

I have built a web application to detect breast cancer from ultrasound images. It takes as input an image with format .png or .jpg.
After the 'Detect Breast Cancer' button is pressed, it returns the test image with the predicted bounding box.

## Article

The article with the explanations is [Building a Web Application to detect Breast Cancer in Ultrasound images]().

## Tools used in the project

* [Datature](https://www.datature.io/)
* [Streamlit](https://streamlit.io/)

## Project Structure

* ```input/```: Some test images for prediction
* ```output/```: Output folder to store predicted images
* ```requirements.txt```: Python dependencies
* ```saved_model/```: Artifact exported from Datature's platform
* ```model_architecture/```: contains the following scripts 
  * ```predict.py```: Python script to make prediction on new images
  * ```app.py```: Python script to build the web application
  
