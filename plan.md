we are going to build a tool that will help users to run high quality particle image velocimetry, PIV. the tool will use openpiv and some free and open llm model to do the following: 
1. user uploads two images and wants to get very good PIV result. 
2. we do not know the correct set of parameters for openpiv processing: number of steps, window size per step, overlap per step, outlier filtering using signal to noise threshold and so on. so the tool will take the two images, learn about their image construction: brightness, uniformity of their background, light, number and size of particle images, contrast, etc. 
3. Then the agent will run openpiv-python on the simplest two or three cases with a single pass of 64 or 32 pixels and get some approximate measure of mean velocity or velocity ranges. 
4. Then the agent will take the first image and deflect it by a ground truth flow of the same dynamic range of velocities and the same mean velocity magnitude. The ground truth will be an analog set of Taylor Green vortices and shear. The agent will create the new second image that resembles the previous one, but with a known flow field. 
5. At this stage the agent will apply openpiv.windef with some optimization of the parameters until the result is closest to the ground truth, with the minimal number of outliers, and lowest RMSE 
6. Then the same parameters will be used for another openpiv-python run with windef to get what we believe is the best optimal set of parameters. 

We work in Python, and for easy prototyping, we use Marimo notebooks. 