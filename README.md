# VectorNST

<div align="center">
  <img src="./images/owl_stylized_owl1.jpg" alt="img1" width="512" height="160"/>
  <img src="./images/scene6_stylized_scene8.jpg" alt="img2" width="512" height="130"/>
  <img src="./images/flower.jpg" alt="img3" width="512" height="160"/>
</div>

The official implementation of VectorNST (vector neural style transfer) model in PyTorch.

[Demo service](http://81.3.154.178:5001/vector_style_transfer).

## Usage

### Requirements

Install these requirements:

* [PyTorch](https://pytorch.org) and Torchvision
* Pillow
* NumPy
* [DiffVG](https://github.com/IzhanVarsky/diffvg) and its requirements

Example steps of installing:

* `pip install torch torchvision pillow numpy`
* `pip install cssutils scikit-learn scikit-image svgwrite svgpathtools matplotlib`
* `git clone --recursive https://github.com/IzhanVarsky/diffvg`
* `cd diffvg && python ./setup.py install && cd ..`

### Testing

Run our example `python test_vector_nst.py --content_img ./images/owl.svg --style_img ./images/owl1.jpg`.

You can also specify some other parameters: run `python test_vector_nst.py --help` to see more info.
