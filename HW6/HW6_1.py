import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import os

# Step 1: 構建預訓練的 VGG16 模型並添加自定義層
def build_model(num_classes=2):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False  # 冻结预训练层

    x = Flatten()(base_model.output)
    x = Dense(128, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Step 2: 訓練模型（此處假設您已經準備好醫療口罩的數據集）
def train_model(model, train_dir, val_dir):
    datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest")

    
    train_generator = datagen.flow_from_directory(train_dir, 
                                                  target_size=(224, 224), 
                                                  batch_size=32, 
                                                  class_mode="categorical")
    val_generator = datagen.flow_from_directory(val_dir, 
                                                target_size=(224, 224), 
                                                batch_size=32, 
                                                class_mode="categorical")

    model.fit(train_generator, validation_data=val_generator, epochs=10)

# Step 3: 從 URL 載入圖像並進行分類
def classify_image(model, image_url, class_names):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    print(f"Predicted class: {class_names[predicted_class]}")

# 主函數
if __name__ == "__main__":
    # 構建模型
    model = build_model(num_classes=2)

    # 訓練模型（設置訓練和驗證數據的路徑）
    train_dir = "./data/train"  # 替換為您的訓練數據目錄
    val_dir = "./data/val"      # 替換為您的驗證數據目錄
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_model(model, train_dir, val_dir)

    # 測試圖像分類
    image_url = input("請輸入圖像的 URL: ")
    class_names = ["No Mask", "Mask"]  # 替換為您的分類名稱
    classify_image(model, image_url, class_names)
from sklearn.metrics import confusion_matrix, classification_report

# 測試數據生成器
test_generator = datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

# 預測結果
predictions = model.predict(test_generator)
y_true = test_generator.classes
y_pred = np.argmax(predictions, axis=1)

# 混淆矩陣與分類報告
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
