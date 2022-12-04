# Web Application to detect Breast Cancer in Ultrasound images

I have built a web application to detect breast cancer from ultrasound images. It takes as input an image with format .png or .jpg.
After the 'Detect Breast Cancer' button is pressed, it returns the test image with the predicted bounding box.

## Article

The article with the explanations is [Building a Web Application to detect Breast Cancer in Ultrasound images](https://medium.com/mlearning-ai/building-a-web-application-to-detect-breast-cancer-in-ultrasound-images-df391483fbd9?sk=0718b6dfc0475bbab62c354288207027).

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
  
## Run the application with the following command line

```streamlit run app.py --input "../input" --output "../output" --width 640 --height 640 --threshold 0.7 --model "../saved_model" --label "../label_map.pbtxt"```

where:
* ```input``` and ```output``` correspond to the path of the input images and path of the produced images by the web app
* width and height need to be equal to 640 to apply the model on the test images
* ```model``` and ```label`` correspond to the path of the saved artifact and the path of the label map
