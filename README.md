# Label-Free Quantification of Gold Nanoparticles using MC-CNN

Using Deep Learning to count the number of AuNPs in a live cell image.

## Requirements and Setup
Python 3.9 and Pytorch 2.1.2+cu118 have been used.

See `requirements.txt` for other python libraries used.

To install, set up a virtual environment using pip. Then install required libraries using
```shell
pip install -r requirements.txt
```

### Data
Please contact the authors for access to the data used for training the models.

### Models

You can download the [MC-CNN model](https://drive.google.com/file/d/1tkbzfbDVlR2rvheVKT8jPHaY3hQJdVKK/view?usp=sharing) here. If the link does not work, please contact the authors. 

### Count AuNPs
Put images for counting in a folder.

In `scripts/asmm-cell-npns/infer.py`, change the `model path` in Line 17 to where you saved the above model.

Then change the `image path` in Line 30 to the path to the folder where images to count are stored.

Finally, run the script

```shell
python scripts/asmm-cell-npns/infer.py run
```

### Citation

If you use any of the code, trained models or data, please cite the paper.

> Mohsin, A. S. M. and Choudhury, S. H. (2024). "A label free quantification of gold nanoparticle at the single cell level using multi-column convolutional neural network (MC-CNN)," *Currently in Review*
