import gradio as gr
import tensorflow as tf

model = tf.keras.models.load_model('farmsense_model.h5', compile=False)

class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

def predict(img):
    img = tf.image.resize(img, (128, 128))
    img = tf.expand_dims(img, 0)
    preds = model.predict(img)[0]
    return {class_names[i]: float(preds[i]) for i in range(len(class_names))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload a leaf photo"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="FarmSense AI - Crop Disease Detection"
)

if __name__ == "__main__":
    demo.launch()