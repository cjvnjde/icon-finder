import tf from "@tensorflow/tfjs-node";
import {L2NormalizationLayer} from "./L2NormalizationLayer.js";


// Register the custom layer with TensorFlow.js's serialization system.
// This allows the model to be saved and loaded correctly.

export function createModel(inputShape, embeddingDim = 128) {
    const model = tf.sequential();

    // ... (Input, Conv, Pooling, and Dense layers remain the same as the previous fix) ...
    model.add(tf.layers.inputLayer({inputShape}));

    model.add(tf.layers.conv2d({
        filters: 16,
        kernelSize: 5,
        activation: "relu",
        padding: "same",
        kernelInitializer: "heNormal"
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: "relu",
        padding: "same",
        kernelInitializer: "heNormal"
    }));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: "relu",
        padding: "same",
        kernelInitializer: "heNormal"
    }));
    model.add(tf.layers.batchNormalization());

    model.add(tf.layers.globalAveragePooling2d({}));

    model.add(tf.layers.dense({units: 256, activation: "relu", kernelInitializer: "heNormal"}));
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dropout({rate: 0.4}));

    model.add(tf.layers.dense({
        units: embeddingDim,
        activation: "linear",
        name: "embedding",
        kernelInitializer: "glorotUniform"
    }));

    // --- CORRECTION: Use the custom L2 normalization layer ---
    model.add(new L2NormalizationLayer());

    return model;
}
