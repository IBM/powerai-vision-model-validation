# PowerAI Vision model validation

Use a deployed model API endpoint to classify images and compare the inference results to "ground truth".

Running the provided Jupyter notebook with your own ground truth images and your deployed model endpoint will generate charts and statistics to demonstrate the accuracy of your model.

## Background

With PowerAI Vision, trained models include testing and accuracy information, but there is often a need to continue to validate with additional test images.

The code provided here, allows an external dataset, with known ground truth values, to be evaluated.

A variety of common model evaluation statistics are produced to represent the accuracy of the model based on this test.

## Results

The Jupyter notebook output provides graphical and tabular output.  In addition, the statistics are saved as CSV files for use with other tools or reports.

## Limitations

This currently only applies to image classification and not object detection.
