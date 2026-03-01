import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _(mo):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from scipy.ndimage import map_coordinates
    from openpiv import pyprocess, validation, filters, scaling, windef
    import json
    # Assuming you use a local open model via Ollama or an open API via LiteLLM
    import litellm

    def upload_and_analyze():
        # Step 1: UI for Image Upload
        upload_img1 = mo.ui.file(label="Upload Image A")
        upload_img2 = mo.ui.file(label="Upload Image B")
    
        # Wait for uploads...
        if upload_img1.value and upload_img2.value:
            # Load images
            img_a = cv2.imdecode(np.frombuffer(upload_img1.value[0].contents, np.uint8), 0)
            img_b = cv2.imdecode(np.frombuffer(upload_img2.value[0].contents, np.uint8), 0)
        
            # Step 2: Image Analysis
            # Calculate background, particle density, contrast, etc.
            mean_brightness = np.mean(img_a)
            contrast = np.std(img_a)
        
            # Simple particle estimation (thresholding)
            _, thresh = cv2.threshold(img_a, mean_brightness + contrast, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
            # Filter background label
            particle_areas = stats[1:, cv2.CC_STAT_AREA]
            avg_particle_size = np.mean(particle_areas) if len(particle_areas) > 0 else 0
            particle_density = len(particle_areas) / (img_a.shape[0] * img_a.shape[1])
        
            image_stats = {
                "mean_brightness": float(mean_brightness),
                "contrast": float(contrast),
                "avg_particle_size_pixels": float(avg_particle_size),
                "particle_density": float(particle_density)
            }
        
        return upload_img1, upload_img2, img_a, img_b, image_stats

    def baseline_piv(img_a, img_b):
        # Step 3: Run a single pass 64x64 or 32x32 to get dynamic range
        window_size = 64
        overlap = 32
        search_area = 64
    
        u, v, sig2noise = pyprocess.extended_search_area_piv(
            img_a.astype(np.int32), img_b.astype(np.int32), 
            window_size=window_size, overlap=overlap, 
            dt=1.0, search_area_size=search_area, sig2noise_method='peak2peak'
        )
    
        # Estimate mean velocity and max displacement
        v_mag = np.sqrt(u**2 + v**2)
        mean_vel = np.nanmean(v_mag)
        max_vel = np.nanmax(v_mag)
    
        velocity_stats = {
            "mean_velocity_px": float(mean_vel),
            "max_velocity_px": float(max_vel)
        }
    
        return velocity_stats, u, v

    def create_synthetic_ground_truth(img_a, velocity_stats):
        # Step 4: Taylor Green vortices + shear
        h, w = img_a.shape
        Y, X = np.mgrid[0:h, 0:w]
    
        # Scale parameters based on dynamic range
        A = velocity_stats["max_velocity_px"] * 0.8  # Amplitude
        L = min(h, w) / 4.0  # Vortex size scale
        shear_rate = velocity_stats["mean_velocity_px"] / h
    
        # Ground truth flow field
        u_gt = A * np.cos(X/L) * np.sin(Y/L) + (shear_rate * Y)
        v_gt = -A * np.sin(X/L) * np.cos(Y/L)
    
        # Warp Image A to create Synthetic Image B using backward mapping
        coords = np.array([Y - v_gt, X - u_gt])
        img_b_synth = map_coordinates(img_a, coords, order=1, mode='reflect')
    
        # Add slight synthetic noise to simulate real-world conditions
        noise = np.random.normal(0, 2, img_b_synth.shape)
        img_b_synth = np.clip(img_b_synth + noise, 0, 255).astype(np.uint8)
    
        return u_gt, v_gt, img_b_synth

    def optimize_parameters_with_llm(img_a, img_b_synth, u_gt, v_gt, image_stats, velocity_stats):
        # Setup openpiv.windef settings generator
        def run_windef_eval(settings_dict):
            # Apply dictionary parameters to a Windef Settings object
            settings = windef.Settings()
            settings.windowsizes = settings_dict["windowsizes"]
            settings.overlap = settings_dict["overlap"]
            settings.sig2noise_threshold = settings_dict["sig2noise_threshold"]
            # ... map other parameters
        
            # Run windef on synthetic pair
            x, y, u, v, s2n = windef.piv(img_a, img_b_synth, settings)
        
            # Validate to count outliers
            u_val, v_val, mask = validation.sig2noise_val(u, v, s2n, threshold=settings.sig2noise_threshold)
            outlier_ratio = np.sum(mask) / mask.size
        
            # Calculate RMSE against ground truth (interpolating GT to PIV grid)
            u_gt_sampled = u_gt[y.astype(int), x.astype(int)]
            v_gt_sampled = v_gt[y.astype(int), x.astype(int)]
        
            rmse = np.sqrt(np.nanmean((u_val - u_gt_sampled)**2 + (v_val - v_gt_sampled)**2))
            return rmse, outlier_ratio

        # LLM Agent loop
        history =[]
        best_params = None
        best_rmse = float('inf')
    
        # Starting parameters
        current_params = {
            "windowsizes": [64, 32],
            "overlap": [32, 16],
            "sig2noise_threshold": 1.2
        }
    
        system_prompt = """
        You are an expert in Particle Image Velocimetry (PIV). 
        Your goal is to optimize openpiv parameters to minimize RMSE and Outlier Ratio.
        Output ONLY a JSON containing 'windowsizes' (list of ints), 'overlap' (list of ints), and 'sig2noise_threshold' (float).
        """

        for iteration in range(3): # Run 3-5 optimization steps
            rmse, outliers = run_windef_eval(current_params)
            history.append({"params": current_params, "rmse": rmse, "outliers": outliers})
        
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = current_params
            
            # Ask LLM for next parameters
            prompt = f"""
            Image Stats: {image_stats}
            Velocity Stats: {velocity_stats}
            History of trials: {json.dumps(history)}
            Based on the history, suggest the next set of parameters to reduce RMSE.
            """
        
            # Call open LLM (e.g., Llama-3 via litellm or local Ollama)
            response = litellm.completion(
                model="ollama/llama3", # or "groq/llama3-70b-8192", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )
        
            current_params = json.loads(response.choices[0].message.content)
        
        return best_params, history

    def final_piv_run(img_a, img_b, best_params):
        settings = windef.Settings()
        settings.windowsizes = best_params["windowsizes"]
        settings.overlap = best_params["overlap"]
        settings.sig2noise_threshold = best_params["sig2noise_threshold"]
    
        # Run on original user images
        x, y, u, v, s2n = windef.piv(img_a, img_b, settings)
    
        # Filter and replace outliers
        u, v, mask = validation.sig2noise_val(u, v, s2n, threshold=settings.sig2noise_threshold)
        u, v = filters.replace_outliers(u, v, method='localmean', max_iter=3, kernel_size=3)
    
        # Plotting code using matplotlib...
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.quiver(x, y, u, v, color='r')
        ax.imshow(img_a, cmap='gray')
        ax.set_title("Final Optimized PIV Flow Field")
    
        return fig

    return


if __name__ == "__main__":
    app.run()
