## Prerequisite
`flask` is required.

To install, run:

`pip install flask`

## Run
`python run.py`

This will pop up a webpage directing to a localhost, e.g. `http://127.0.0.1:5000/`. 

You can then navigate the samples (according to order in directory) by clicking the self explained button. 

To view a specific sample, append the id of the sample to the path of the webpage.
### Example
To view the sample `76532309b`, navigate `http://127.0.0.1:5000/76532309b`.

## Customize
Put your images to be visualized into the `data` subdirectory of this project.

Modify the path to the image directory in `./template/imshow.html`

```
<h1>Original image ({{path_to_img}})</h1>
<img src=./data/***img***/{{path_to_img}}.jpg />
<h1>Label</h1>
<img src=./data/***mask_colored/test***/{{path_to_img}}.png />
<h1>Prediction</h1>
<img src=./data/***drn_d_22_200_val_color/img***/{{path_to_img}}.png />
```

Edit only the phrases enveloped by *** *** pairs.


