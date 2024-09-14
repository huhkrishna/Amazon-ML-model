import os
import urllib

def download_images(image_links, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    for image_link in image_links:
        filename = os.path.basename(image_link)
        image_save_path = os.path.join(download_folder, filename)

        if os.path.exists(image_save_path):
            print(f"Image already exists: {image_save_path}")
            continue

        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            print(f"Downloaded: {image_save_path}")
        except Exception as e:
            print(f"Failed to download {image_link}: {e}")