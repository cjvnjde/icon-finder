import tf from "@tensorflow/tfjs-node";

/**
 * Create an improved embedding model for sketch-to-icon similarity
 * Alternative version without globalAveragePooling2d
 * @param {number[]} inputShape - Shape of input images [height, width, channels]
 * @param {number} embeddingDim - Dimension of the embedding space
 * @returns {tf.Sequential} The model
 */
export function createModel(inputShape, embeddingDim = 128) {
    const model = tf.sequential();

    model.add(
        tf.layers.inputLayer({
            inputShape,
        }),
    );

    // First conv block - fewer filters for simpler features
    model.add(
        tf.layers.conv2d({
            filters: 16,
            kernelSize: 5,
            activation: "relu",
            padding: "same",
            kernelInitializer: "heNormal",
        }),
    );
    model.add(tf.layers.batchNormalization());
    model.add(
        tf.layers.maxPooling2d({
            poolSize: 2,
            strides: 2,
        }),
    );

    // Second conv block
    model.add(
        tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: "relu",
            padding: "same",
            kernelInitializer: "heNormal",
        }),
    );
    model.add(tf.layers.batchNormalization());
    model.add(
        tf.layers.maxPooling2d({
            poolSize: 2,
            strides: 2,
        }),
    );

    // Third conv block
    model.add(
        tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: "relu",
            padding: "same",
            kernelInitializer: "heNormal",
        }),
    );
    model.add(tf.layers.batchNormalization());

    // Alternative to globalAveragePooling2d: Use regular pooling + flatten
    model.add(
        tf.layers.maxPooling2d({
            poolSize: 2,
            strides: 2,
        }),
    );

    // Flatten for dense layers
    model.add(tf.layers.flatten());

    // Reduced dropout to help learning
    model.add(
        tf.layers.dropout({
            rate: 0.2,
        }),
    );

    // Dense layers
    model.add(
        tf.layers.dense({
            units: 512,
            activation: "relu",
            kernelInitializer: "heNormal",
        }),
    );
    model.add(tf.layers.batchNormalization());
    model.add(
        tf.layers.dropout({
            rate: 0.3,
        }),
    );

    // Embedding layer - we'll handle L2 normalization in the loss function
    model.add(
        tf.layers.dense({
            units: embeddingDim,
            activation: "linear",
            name: "embedding",
            kernelInitializer: "glorotUniform",
        }),
    );

    return model;
}
