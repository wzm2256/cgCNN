1. The exemplar sound texture is placed in ./Sound, and the trained model is placed in ./Saved_model
2. Make sure you can synthesize using the pretrained model.
''' 
python Test.py norm_shaking_paper.wav --step 600
'''
3. The synthesized image texture will be in ./Produce_Generator
4. Train your model using your data.
'''
python Train.py norm_shaking_paper.wav
'''
