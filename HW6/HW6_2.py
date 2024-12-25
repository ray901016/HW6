import os
from transformers import pipeline
from diffusers import StableDiffusionPipeline

# 確保代碼在多進程環境下正常運行（特別是 Windows）
if __name__ == "__main__":
    # 選擇性設置：禁用 Symlink 警告
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    try:
        # Step 1: 翻譯中文 "一個狗和一隻貓" -> 英文
        print("正在執行翻譯...")
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
        chinese_text = "跳舞的黑人"
        translation_result = translator(chinese_text, max_length=40)
        translated_text = translation_result[0]["translation_text"]
        print("翻譯結果:", translated_text)

        # Step 2: 使用翻譯結果生成圖像
        print("正在生成圖像...")
        # 初始化 Stable Diffusion 圖像生成 pipeline
        image_generator = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        image_generator = image_generator

        # 用翻譯結果作為圖像生成的提示
        image = image_generator(translated_text).images[0]

        # 保存生成的圖像
        image.save("generated_image.png")
        print("圖像生成完成，已保存為 'generated_image.png'")

    except Exception as e:
        print("執行過程中發生錯誤:", e)
