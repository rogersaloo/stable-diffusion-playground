# Stab;e Diffusion Playground  

## Description
Mid Journey, Stable Diffusion can be used to generate images. They are a foundation in the generation of images. A playground on <NAME_OF_MODEL> model released on <DATE_OF_RELEASE>.
Checkout the <[OFFICIAL_REPO](here)> or navigate to ```officaial_repo``` folder on the root dotirectory.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

# How Diffusion works
1. You have a set of images and you need more images.
2. YOu take the images you have and then add noises progressively into the image.
![image](assets\1-initial_images.png)
3. Add ddpm and use the sampled images in the process.
4. THe unet is used as the neural network
- image input and output are th e same size
- embedds input then upsamples input
- Takes in additiona information of the time embedding to determine the type of noise 
- context embedding used to enable control in the generation.

## Installation

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

Clone the repository locally and install with

```
git 
cd segment-anything;
pip install -e .
```

Dependancies
```
pip install <DEPENDANCIES>
```

## <a name="GettingStarted"></a>Getting Started

First download a [model checkpoint](model_checkpoints_option). 
Then the model can be used in just a few lines:

```
from library import

model = continue code
predict = predict(continue code)

```


### ONNX Export

How to export model to ONNX:

```
python scripts/export_onnx_model.py --checkpoint <path/to/checkpoint> --model-type <model_type> --output <path/to/output>
```

### Web demo

The `demo/` folder has a simple one page django app which shows how to run mask prediction with the exported ONNX model in a web browser with multithreading.

## <a name="Models"></a>Model Checkpoints

The finetuned model checkpoint are provided as follows

```
from model_name import model_name_registry
sam = model_name_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
```

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ link_tomodel_model.](link.pth)**

## Dataset

See [here](#) for an overview of the datastet. The dataset can be downloaded [here](). 
The dataset contains the following

## <a name="Models"></a>Directory Organization
The repositroy contains the official <MODEL_NAME> repo together with othe implementations on the root directory.


## License

The model is licensed under the [Apache 2.0 license](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

## Resources
1.Images DeepLearning.ai stable diffusion short course
2. Setup cudnn on [ubuntu](https://fizzylogic.nl/2022/11/02/how-to-set-up-cuda-and-pytorch-in-ubuntu-22-04-on-wsl2)
## Citing Original model

If you use <MODEL_NAME> or 

```
@mycitation{}

```
