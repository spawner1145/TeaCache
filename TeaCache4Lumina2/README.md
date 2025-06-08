<!-- ## **TeaCache4LuminaT2X** -->
# TeaCache4Lumina2

[TeaCache](https://github.com/LiewFeng/TeaCache) can speedup [Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0) without much visual quality degradation, in a training-free manner. The following image shows the experimental results of Lumina-Image-2.0 and TeaCache with different versions: Lumina-Image-2.0 (~25 s), TeaCache (0.2) (~16.7 s, 1.5x speedup), TeaCache (0.3) (~15.6 s, 1.6x speedup), TeaCache (0.5) (~13.79 s, 1.8x speedup), TeaCache (1.1) (~11.9 s, 2.1x speedup).

<p align="center">
    <img src="https://github.com/user-attachments/assets/aea9907b-830e-497b-b968-aaeef463c7ef" width="150" style="margin: 5px;">
    <img src="https://github.com/user-attachments/assets/0e258295-eaaa-49ce-b16f-bba7f7ada6c1" width="150" style="margin: 5px;">
    <img src="https://github.com/user-attachments/assets/44600f22-3fd4-4bc4-ab00-29b0ed023d6d" width="150" style="margin: 5px;">
    <img src="https://github.com/user-attachments/assets/bcb926ab-95fd-4c83-8b46-f72581a3359e" width="150" style="margin: 5px;">
    <img src="https://github.com/user-attachments/assets/ec8db28e-0f9b-4d56-9096-fdc8b3c20f4b" width="150" style="margin: 5px;">
</p>

## 📈 Inference Latency Comparisons on a single 4090 (step 50)


|      Lumina-Image-2.0      |        TeaCache (0.2)       |    TeaCache (0.3)    |     TeaCache (0.5)    |     TeaCache (1.1)    |
|:-------------------------:|:---------------------------:|:--------------------:|:---------------------:|:---------------------:|
|         ~25 s             |        ~16.7 s                |     ~15.6 s            |       ~13.79 s             |       ~11.9 s             |

## Installation

```shell
pip install --upgrade diffusers[torch] transformers protobuf tokenizers sentencepiece
pip install flash-attn --no-build-isolation
```

## Usage

You can modify the thresh in line 154 to obtain your desired trade-off between latency and visul quality. For single-gpu inference, you can use the following command:

```bash
python teacache_lumina2.py
```

## Citation
If you find TeaCache is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```
@article{liu2024timestep,
  title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
  author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
  journal={arXiv preprint arXiv:2411.19108},
  year={2024}
}
```

## Acknowledgements

We would like to thank the contributors to the [Lumina-Image-2.0](https://github.com/Alpha-VLLM/Lumina-Image-2.0) and [Diffusers](https://github.com/huggingface/diffusers).
