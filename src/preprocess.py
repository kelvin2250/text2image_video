import os
from datasets import load_dataset
from tqdm import tqdm

def download_flickr30k_all(
    output_dir="data",
    image_folder="flickr30k_images",
    caption_file="captions.txt"
):
    image_dir = os.path.join(output_dir, image_folder)
    os.makedirs(image_dir, exist_ok=True)
    caption_path = os.path.join(output_dir, caption_file)

    print("🔹 Downloading full Flickr30K dataset (~30K images)...")
    dataset = load_dataset(
        "nlphuji/flickr30k",
        split="test",
        cache_dir="D:/huggingface_cache"
    )

    with open(caption_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(dataset, desc="Saving images")):
            try:
                captions = item["caption"][:5]
                image = item["image"]
                image_filename = f"img_{i:05d}.jpg"
                image_path = os.path.join(image_dir, image_filename)
                image_rel_path = os.path.join(image_folder, image_filename)

                image.save(image_path)
                for caption in captions:
                    f.write(f"{image_rel_path}\t{caption.strip()}\n")

            except Exception as e:
                print(f"❌ Error saving image {i}: {e}")
                continue


if __name__ == "__main__":
    download_flickr30k_all(
        output_dir="data",
        image_folder="flickr30k_images",
        caption_file="captions.txt"
    )
