# Conditional Generative ConvNets for Exemplar-based Texture Synthesis

This is a tensorflow implement of the cgCNN model proposed in [Conditional Generative ConvNets for Exemplar-based Texture Synthesis](https://arxiv.org/abs/1912.07971).

This repository contains implements of three tasks as described below. 

|Tasks|Description|
|-----|-----|
|Texture synthesis|Synthesize new textures that are visually similar to the given exemplar texture.|
|Texture expansion|Synthesis new textures that are larger (or arbitrarily large) than the given exemplar texture.|
|Texture inpainting|Fill the corrupted region in the given exemplar texture.|

Three types of textures, i.e. dynamic, image and sound textures, are considered for each task, and each applications is placed in an individual folder. For example, the code for dynamic texture inpainting is placed in *./dynamic_inpaint*, and the code for image texture expansion is placed in *./image_expansion*.

Please visit our [project page](http://captain.whu.edu.cn/cgcnn-texture/) for more results.

## Requirements
- python=3.5
- tensorflow=1.8.0
- keras=2.1.6
- librosa  (for audio loading and saving)
- ffmpeg (for video loading and saving)

## Results
### Texture synthesis
#### Image texture synthesis

|Exemplar|Result|
|--------------|--------------|
| <img src="readme_fig\i_s\bubbly_0038.jpg" width="128"/>  | <img src="readme_fig\i_s\bubbly_0038.jpg_inner_10__layer_S_3_layer_D_9_IsMean_0_Adam_0_2_5000_.jpg" width="128"/> |
| <img src="readme_fig\i_s\Texture54.png" width="128"/>  | <img src="readme_fig\i_s\Texture54.png_inner_50_layer_S_3_layer_D_9_IsMean_0_Adam_0_1_3000_.jpg" width="128"/> |


#### Dynamic texture synthesis
|Exemplar|Result|
|--------------|--------------|
|<img src="readme_fig\d_s\sample_o.gif">|<img src="readme_fig\d_s\sample.gif"> |

#### Sound texture synthesis
|Exemplar|Result|
|--------------|--------------|
|[applause_exemplar](readme_fig/s_s/norm_Enthusiastic_applause.wav)|[applause_synthesis](readme_fig/s_s/norm_Enthusiastic_applause.wav_Model_1_depth_4_IsMean_0_Adam_0_Fou_0_Gau_1e-08_inner_10_1_3000_.wav) |

### Texture expansion
#### Image texture expansion

|Exemplar|Result|
|--------------|--------------|
| <img src="readme_fig\i_e\peppers.jpg" width="128">  | <img src="readme_fig\i_e\peppers.jpg_layer_S_3_layer_D_9_inner_10_IsMean_0_Adam_1_normal_False_Gau_0.0_Fou_0.0_diversity_No_d_weight_0_0_step_2200_.jpg" width="448"> |

#### Dynamic texture expansion
|Exemplar|Result|
|--------------|--------------|
|<img src="readme_fig\d_e\origin.gif" width="128">|<img src="readme_fig\d_e\expansion.gif" width="448"> |

#### Sound texture expansion
|Exemplar|Result|
|--------------|--------------|
|[shaking_paper_exemplar](readme_fig/s_e/norm_shaking_paper.wav)|[shaking_paper_expansion](readme_fig/s_e/norm_shaking_paper.wav_depth_4_inner_10_IsMean_0_Adam_0_Gau_0.0_diversity_No_d_weight_0.wav) |

### Texture inpainting
#### Image texture inpainting
|Exemplar|Result|
|--------------|--------------|
| <img src="readme_fig\i_i\masked_fibrous_0145.jpg" width="128"/>  | <img src="readme_fig\i_i\fibrous_0145.jpg_mask2.png_mean_1_inner_10_tv_0.0_fou_0.0001900_.jpg" width="128"/> |

#### Dynamic texture inpainting
|Exemplar|Result|
|--------------|--------------|
| <img src="readme_fig\d_i\sample_o.gif" width="128"/>  | <img src="readme_fig\d_i\sample.gif" width="128"/> |

#### Sound texture inpainting
|Exemplar|Result|
|--------------|--------------|
|[bees_exemplar](readme_fig/s_i/masked_norm_Bees.wav)|[bees_inpainted](readme_fig/s_i/norm_Bees.wav__mean_1_inner_10_fou_1600_.wav) |

## Reference

    @misc{wang2019conditional,
    title={Conditional Generative ConvNets for Exemplar-based Texture Synthesis},
    author={Zi-Ming Wang and Meng-Han Li and Gui-Song Xia},
    year={2019},
    eprint={1912.07971},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }

For any question, please contact Ziming Wang (wangzm@whu.edu.cn).
