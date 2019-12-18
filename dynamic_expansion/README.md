1. The exemplar dynamic texture is placed in *./Image*, and the trained model is placed in *./Saved_model*
2. Make sure you can synthesize textures using the pretrained model.
```
python Test.py VIDEO/water --step 1500 --size_t 48 --size_s 512
```
3. The synthesized image texture will be in *./Produce_Generator*. 

    **Notice:** The output image will be 64 pixels smaller than the given spatial size parameter, and 4 frames smaller than the given temporal size parameter, as we crop the border pixels (frames) for better quality.

4. Train your model using your data.
```
python Train.py VIDEO/water
```
5. The trained model will be in *./Saved_model*