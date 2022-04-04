# Improving generalization of Batch Whitening by Convolutional Unit Optimization

**Yooshin Cho, Hanbyel Cho, Youngsoo Kim, and Junmo Kim** | [Paper](https://arxiv.org/abs/2108.10629)

**This is the official repository of "Improving generalization of Batch Whitening by Convolutional Unit Optimization", ICCV 2021.**

## Abstract

Batch Whitening is a technique that accelerates and stabilizes training by transforming input features to have a zero mean (Centering) and a unit variance (Scaling), and by removing linear correlation between channels (Decorrelation). In commonly used structures, which are empirically optimized with Batch Normalization, the normalization layer appears between convolution and activation function. Following Batch Whitening studies have employed the same structure without further analysis; even Batch Whitening was analyzed on the premise that the input of a linear layer is whitened. To bridge the gap, we propose a new Convolutional Unit that is in line with the theory, and our method generally improves the performance of Batch Whitening. Moreover, we show the inefficacy of the original Convolutional Unit by investigating rank and correlation of features. As our method is employable off-the-shelf whitening modules, we use Iterative Normalization (IterNorm), the state-of-the-art whitening module, and obtain significantly improved performance on five image classification datasets: CIFAR-10, CIFAR-100, CUB-200-2011, Stanford Dogs, and ImageNet. Notably, we verify that our method improves stability and performance of whitening when using large learning rate, group size, and iteration number.


## Dependencies
Make sure you have the following dependencies installed before proceeding:
- Python 3+ distribution
- PyTorch == 0.4.0

## License
This project is distributed under [MIT license](LICENSE.md). If you use our code/models in your research, please cite our paper:
```
@inproceedings{cho2021improving,
    title={Improving Generalization of Batch Whitening by Convolutional Unit Optimization},
    author={Cho, Yooshin and Cho, Hanbyel and Kim, Youngsoo and Kim, Junmo},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={5321--5329},
    year={2021}
}
```

## Acknowledgement
Part of our code is borrowed from [shiftresnet-cifar](https://github.com/alvinwan/shiftresnet-cifar) and [IterNorm-pytorch](https://github.com/huangleiBuaa/IterNorm-pytorch). Please refer to their project page for further information.
