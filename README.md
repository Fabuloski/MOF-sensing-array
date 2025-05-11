# Arrays of Bimetallic Metallophthalocyanine-based Conductive Metal–Organic  Framework Materials Chemiresistive Detection and Differentiation of Toxic Gases.
:rocket: This repo contains data and code to reproduce the results for:
> Evan Cline, Hyuk-Jun Noh, Georganna Benedetto, Gbenga Fabusola, Simon Cory, and Katherine A. Mirica, "Arrays of Bimetallic Metallophthalocyanine-based Conductive Metal–Organic  Framework Materials Chemiresistive Detection and Differentiation of Toxic Gases."

We describe the sequence of steps we took to make our paper reproducible. The output of each step is saved as a file, you can start at any step.

## required software
required software/packages:
* [Python 3](https://www.python.org/downloads/) version 3.8 or newer
* [Marimo Notebook](https://docs.marimo.io/)

## the sensor array response dataset
We obtained the dataset of sensors' responses to analytes from an experimental collaboration at Dartmouth College led by Dr. Katherine A. Mirica.

## analysis
We run the PCA and supervised learning on the sensor array dataset using `sensor_response.py`.

## overview of directories
- `data`: contains Microsoft Excel raw response files of sensors to different gas concentrations [ppm]
- `responses`: contains visualization of the response of sensors to every single experiment (ppm of gas), with the features extracted

