For other model weights you can go to : https://drive.google.com/drive/folders/1uw-fqrLzP6qV6-7KREq0RVaaDF4Et2rs?usp=drive_link
For downloading dataset:https://drive.google.com/file/d/1qgEvdtpovP3PYcoJaS_ALqyTezVNqVlb/view?usp=drive_link
For local collab runtim : https://colab.research.google.com/drive/1TlqGNqFCyOV6RaFaqIhNT-wuT0cQz1zE?usp=sharing
For any errors or suggestions feel free to reach us on : aniketkumar02xy@gmail.com,ayonaryan5@gmail.com


Fine-Tuned ESRGAN for Photorealistic Face Restoration

This project is the result of a rigorous fine-tuning process on the RealESRGAN_x4plus model. The goal was to create a "specialist" model that excels at 4x upscaling and restoring fine details in human faces, surpassing the performance of the original "generalist" model on this specific task.

The final model, net_g_2000.pth (from the FaceESRGAN_finetune_20k_v1 run), represents the scientifically-validated "peak" of quality found during experimentation, achieving a PSNR of 33.95 on our validation set.

Final Model: Generalist vs. Specialist

The fine-tuning process measurably improved the model's ability to render realistic textures on faces. The original model is a "jack-of-all-trades," while our model is a "master" of faces.

Note: You should upload your own Input, Pretrained, and Ours comparison images to your repo and update the paths in your README to display them.

How to Use This Model (Demo in Google Colab)

You can easily run our champion model on any image.

1. Upload Your Model

First, upload your champion model file (net_g_2000.pth) to your GitHub repository.

2. Open in Google Colab

Open a new Google Colab notebook, connect to a GPU runtime, and run the following three cells.

Cell 1: Setup (Clone Repos & Install)

%cd /content
print("Cloning repositories...")
!git clone [https://github.com/xinntao/Real-ESRGAN.git](https://github.com/xinntao/Real-ESRGAN.git)
!git clone [https://github.com/xinntao/basicsr.git](https://github.com/xinntao/basicsr.git)

print("Installing basicsr dependencies...")
%cd /content/basicsr
!pip install -r requirements.txt
!python setup.py develop

print("Installing Real-ESRGAN dependencies...")
%cd /content/Real-ESRGAN
!pip install -r requirements.txt
!python setup.py develop

print("Setup complete! Ready for GPU inference.")


Cell 2: Download Your Fine-Tuned Model

print("Downloading your fine-tuned champion model...")
!mkdir -p /content/Real-ESRGAN/models

# IMPORTANT: Replace this URL with the "raw" download link to your model file
!wget -O /content/Real-ESRGAN/models/my_champion_model.pth "[https://github.com/YOUR_USERNAME/YOUR_REPO/raw/main/net_g_2000.pth](https://github.com/YOUR_USERNAME/YOUR_REPO/raw/main/net_g_2000.pth)"

print("Model downloaded!")


Cell 3: The "Degrade & Enhance" Demo
This cell lets you upload any high-resolution image, degrades it, runs your model on it, and shows a side-by-side comparison.

#@title Degrade, Enhance & Compare Demo
#@markdown ---
#@markdown ### 1. Upload your HR image to the `/content/` folder
#@markdown ### 2. Enter its *exact* filename here:

high_res_filename = "my_uploaded_image.png" #@param {type:"string"}

#@markdown ---
#@markdown ### 3. Click the "Play" button to run the demo.

# --- Import necessary libraries ---
import os, glob, cv2, time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Image

# --- 4. CONFIGURE ALL PATHS ---
INPUT_HR_FILE = f"/content/{high_res_filename}"
TEMP_LR_FILE_PATH = "/content/temp_lr_image.png"
OUTPUT_FOLDER = "/content/temp_demo_results"
FINAL_OUTPUT_FILE_PATH = f"{OUTPUT_FOLDER}/temp_lr_image_out.png"
MODEL_PATH = "/content/Real-ESRGAN/models/my_champion_model.pth"
INFERENCE_SCRIPT_PATH = "/content/Real-ESRGAN/inference_realesrgan.py"

# --- 5. CHECK IF ALL FILES/FOLDERS ARE READY ---
if not os.path.exists(INFERENCE_SCRIPT_PATH):
    print("ERROR: 'inference_realesrgan.py' not found. Please run Cell 2 (Setup).")
elif not os.path.exists(INPUT_HR_FILE):
    print(f"ERROR: File not found: {INPUT_HR_FILE}. Please upload it.")
elif not os.path.exists(MODEL_PATH):
    print(f"ERROR: Champion model not found at: {MODEL_PATH}. Please run Cell 2 (Download).")
else:
    # --- 6. ACTION 1: BICUBIC DEGRADATION ---
    print(f"Degrading '{high_res_filename}' 4x using bicubic interpolation...")
    scale = 4
    img = cv2.imread(INPUT_HR_FILE)
    if img is None:
        print(f"ERROR: Could not read {INPUT_HR_FILE}.")
    else:
        h, w = img.shape[:2]
        h, w = (h // scale) * scale, (w // scale) * scale
        img = img[0:h, 0:w]
        lr_img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(TEMP_LR_FILE_PATH, lr_img)
        print(f"Created temporary 4x low-res image.")

        # --- 7. ACTION 2: RUN ENHANCEMENT ---
        print(f"Running enhancement with your CHAMPION model...")
        start_time = time.time()
        !mkdir -p {OUTPUT_FOLDER}
        !python {INFERENCE_SCRIPT_PATH} \
            -i {TEMP_LR_FILE_PATH} \
            -o {OUTPUT_FOLDER} \
            --outscale 4 \
            --model_path {MODEL_PATH}
        end_time = time.time()
        print(f"Inference complete in {end_time - start_time:.2f} seconds!")

        # --- 8. ACTION 3: DISPLAY SIDE-BY-SIDE ---
        print("Loading images for comparison...")
        if not os.path.exists(FINAL_OUTPUT_FILE_PATH):
            print(f"ERROR: Output file not found. Inference may have failed.")
        else:
            try:
                img_in = mpimg.imread(TEMP_LR_FILE_PATH)
                img_out = mpimg.imread(FINAL_OUTPUT_FILE_PATH)
                print("Displaying comparison...")
                fig, ax = plt.subplots(1, 2, figsize=(18, 9)) 
                
                ax[0].imshow(img_in)
                ax[0].set_title(f'Input: (4x Bicubic Degraded)\n({img_in.shape[0]}x{img_in.shape[1]})', fontsize=14)
                ax[0].axis('off')

                ax[1].imshow(img_out)
                ax[1].set_title(f'Output: (Restored by your Model)\n({img_out.shape[0]}x{img_out.shape[1]})', fontsize=14)
                ax[1].axis('off')

                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"An error occurred while trying to display the images: {e}")


The Training Journey

This model was the result of multiple iterative training runs, totaling over 20,000 iterations, to find the "peak" quality before the model began to overfit.

Base Model: RealESRGAN_x4plus.pth

Dataset: A custom set of 3,143 high-resolution face images, split into 3,043 for training and 100 for validation.

Method: We used a "seeding" strategy. The final 10,000-iteration run (FaceESRGAN_finetune_20k_v1) was "seeded" with the weights of a previously trained 9,000-iteration model. This allowed the model to continue refining its quality from an already high baseline.

Finding the Peak: Why net_g_2000.pth?

The validation logs provided a clear "stop" signal. Training a GAN for too long causes it to overfit, resulting in worse quality on new images. We found the perfect balance.

Validation PSNR (Higher is Better):

@ 2000 iters: 33.9532 (All-Time Peak Quality)

@ 4000 iters: 32.8771 (Overfitting "dip")

@ 6000 iters: 33.4921 (Recovering)

@ 8000 iters: 33.5488 (Recovering)

@ 10000 iters: 33.6256 (Never beat the 2k peak)

The logs scientifically prove that the net_g_2000.pth model from the final run is the champion, possessing the best balance of sharpness and realism before overfitting began.

Acknowledgments

This project is a fine-tuning of the original Real-ESRGAN model. All credit for the base architecture and training framework goes to Xintao Wang et al.

Real-ESRGAN GitHub: https://github.com/xinntao/Real-ESRGAN

ESRGAN Paper: https://arxiv.org/abs/1809.00219
